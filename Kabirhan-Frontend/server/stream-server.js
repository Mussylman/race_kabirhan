/**
 * RTSP to HLS Stream Server
 * 
 * This script converts RTSP camera streams to HLS format for web playback.
 * Requires FFmpeg to be installed on the system.
 * 
 * Usage: node stream-server.js
 */

const { spawn } = require('child_process');
const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 8080;

// Enable CORS for frontend
app.use(cors());

// Serve HLS streams
const STREAMS_DIR = path.join(__dirname, 'streams');
if (!fs.existsSync(STREAMS_DIR)) {
  fs.mkdirSync(STREAMS_DIR, { recursive: true });
}
app.use('/streams', express.static(STREAMS_DIR));

// Camera configurations
const cameras = {
  // PTZ Cameras
  'ptz-1': { rtspUrl: 'rtsp://admin:password@192.168.1.101:554/stream1', name: 'PTZ 1' },
  'ptz-2': { rtspUrl: 'rtsp://admin:password@192.168.1.102:554/stream1', name: 'PTZ 2' },
  'ptz-3': { rtspUrl: 'rtsp://admin:password@192.168.1.103:554/stream1', name: 'PTZ 3' },
  'ptz-4': { rtspUrl: 'rtsp://admin:password@192.168.1.104:554/stream1', name: 'PTZ 4' },
  // Fixed cameras will be added dynamically
};

// Add 25 fixed cameras
for (let i = 0; i < 25; i++) {
  cameras[`fixed-${i}`] = {
    rtspUrl: `rtsp://admin:password@192.168.2.${100 + i}:554/stream1`,
    name: `Fixed ${i + 1}`
  };
}

// Active FFmpeg processes
const activeStreams = new Map();

// Start RTSP to HLS conversion for a camera
function startStream(cameraId) {
  const camera = cameras[cameraId];
  if (!camera) {
    console.error(`Camera ${cameraId} not found`);
    return null;
  }

  if (activeStreams.has(cameraId)) {
    console.log(`Stream ${cameraId} already running`);
    return activeStreams.get(cameraId);
  }

  const streamDir = path.join(STREAMS_DIR, cameraId);
  if (!fs.existsSync(streamDir)) {
    fs.mkdirSync(streamDir, { recursive: true });
  }

  const outputPath = path.join(streamDir, 'index.m3u8');

  // FFmpeg command for RTSP to HLS conversion
  const ffmpegArgs = [
    '-rtsp_transport', 'tcp',           // Use TCP for RTSP (more reliable)
    '-i', camera.rtspUrl,                // Input RTSP stream
    '-c:v', 'libx264',                   // Video codec
    '-preset', 'ultrafast',              // Fast encoding for low latency
    '-tune', 'zerolatency',              // Zero latency tuning
    '-c:a', 'aac',                       // Audio codec
    '-b:a', '128k',                      // Audio bitrate
    '-f', 'hls',                         // Output format
    '-hls_time', '2',                    // Segment duration (seconds)
    '-hls_list_size', '5',               // Number of segments in playlist
    '-hls_flags', 'delete_segments+append_list', // Delete old segments
    '-hls_segment_filename', path.join(streamDir, 'segment_%03d.ts'),
    outputPath
  ];

  console.log(`Starting stream: ${cameraId} (${camera.name})`);
  console.log(`RTSP URL: ${camera.rtspUrl}`);

  const ffmpeg = spawn('ffmpeg', ffmpegArgs);

  ffmpeg.stdout.on('data', (data) => {
    // FFmpeg outputs to stderr, so stdout is usually empty
  });

  ffmpeg.stderr.on('data', (data) => {
    // Log FFmpeg output (can be verbose)
    // console.log(`[${cameraId}] ${data}`);
  });

  ffmpeg.on('close', (code) => {
    console.log(`Stream ${cameraId} closed with code ${code}`);
    activeStreams.delete(cameraId);
    
    // Auto-restart on failure
    if (code !== 0) {
      console.log(`Restarting stream ${cameraId} in 5 seconds...`);
      setTimeout(() => startStream(cameraId), 5000);
    }
  });

  ffmpeg.on('error', (err) => {
    console.error(`Stream ${cameraId} error:`, err.message);
  });

  activeStreams.set(cameraId, {
    process: ffmpeg,
    camera,
    startTime: new Date()
  });

  return activeStreams.get(cameraId);
}

// Stop a stream
function stopStream(cameraId) {
  const stream = activeStreams.get(cameraId);
  if (stream) {
    stream.process.kill('SIGTERM');
    activeStreams.delete(cameraId);
    console.log(`Stopped stream: ${cameraId}`);
    return true;
  }
  return false;
}

// API Endpoints
app.get('/api/cameras', (req, res) => {
  const cameraList = Object.entries(cameras).map(([id, config]) => ({
    id,
    name: config.name,
    rtspUrl: config.rtspUrl,
    hlsUrl: `/streams/${id}/index.m3u8`,
    isActive: activeStreams.has(id)
  }));
  res.json(cameraList);
});

app.post('/api/cameras/:id/start', (req, res) => {
  const { id } = req.params;
  const result = startStream(id);
  if (result) {
    res.json({ success: true, message: `Stream ${id} started`, hlsUrl: `/streams/${id}/index.m3u8` });
  } else {
    res.status(400).json({ success: false, message: `Failed to start stream ${id}` });
  }
});

app.post('/api/cameras/:id/stop', (req, res) => {
  const { id } = req.params;
  const result = stopStream(id);
  if (result) {
    res.json({ success: true, message: `Stream ${id} stopped` });
  } else {
    res.status(400).json({ success: false, message: `Stream ${id} not found` });
  }
});

app.get('/api/streams/status', (req, res) => {
  const status = {};
  activeStreams.forEach((stream, id) => {
    status[id] = {
      name: stream.camera.name,
      running: true,
      startTime: stream.startTime
    };
  });
  res.json(status);
});

// Update camera RTSP URL
app.put('/api/cameras/:id', express.json(), (req, res) => {
  const { id } = req.params;
  const { rtspUrl } = req.body;
  
  if (cameras[id]) {
    cameras[id].rtspUrl = rtspUrl;
    // Restart stream if running
    if (activeStreams.has(id)) {
      stopStream(id);
      startStream(id);
    }
    res.json({ success: true, message: `Camera ${id} updated` });
  } else {
    res.status(404).json({ success: false, message: `Camera ${id} not found` });
  }
});

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    activeStreams: activeStreams.size,
    uptime: process.uptime()
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`
╔════════════════════════════════════════════════════════════╗
║           RTSP to HLS Stream Server                        ║
╠════════════════════════════════════════════════════════════╣
║  Server running on: http://localhost:${PORT}                  ║
║  Streams directory: ${STREAMS_DIR}
║                                                            ║
║  API Endpoints:                                            ║
║  - GET  /api/cameras          - List all cameras           ║
║  - POST /api/cameras/:id/start - Start stream              ║
║  - POST /api/cameras/:id/stop  - Stop stream               ║
║  - GET  /api/streams/status    - Active streams status     ║
║  - PUT  /api/cameras/:id       - Update camera RTSP URL    ║
║                                                            ║
║  HLS Streams: /streams/{camera-id}/index.m3u8              ║
╚════════════════════════════════════════════════════════════╝

Prerequisites:
- FFmpeg must be installed and in PATH
- Camera RTSP URLs must be configured correctly

To start all PTZ cameras:
  curl -X POST http://localhost:${PORT}/api/cameras/ptz-1/start
  curl -X POST http://localhost:${PORT}/api/cameras/ptz-2/start
  curl -X POST http://localhost:${PORT}/api/cameras/ptz-3/start
  curl -X POST http://localhost:${PORT}/api/cameras/ptz-4/start
`);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down...');
  activeStreams.forEach((stream, id) => {
    stream.process.kill('SIGTERM');
  });
  process.exit(0);
});
