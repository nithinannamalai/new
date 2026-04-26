import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

let handLandmarker: HandLandmarker;
let video: HTMLVideoElement;
let outputCanvas: HTMLCanvasElement;
let outputCtx: CanvasRenderingContext2D;

let drawingCanvas: HTMLCanvasElement;
let drawingCtx: CanvasRenderingContext2D;

// State
let drawColor: string | 'rainbow' = '#ef4444'; // default red
let drawThickness = 6;
let bgMode: 'camera' | 'black' | 'white' = 'camera';
let lineStyle: 'normal' | 'straight' | 'dotted' | 'rectangle' | 'circle' = 'normal';

let isDrawing = false;
let prev_x: number | null = null;
let prev_y: number | null = null;
let smooth_x: number | null = null;
let smooth_y: number | null = null;
let start_x: number | null = null;
let start_y: number | null = null;
let last_x: number | null = null;
let last_y: number | null = null;
let last_mid_x: number | null = null;
let last_mid_y: number | null = null;
let lost_frames = 0;
let clear_frames = 0;
const max_lost_frames = 6;
let rainbow_hue = 0;

let undo_stack: ImageData[] = [];
let redo_stack: ImageData[] = [];

// Initialize
async function init() {
  video = document.getElementById("videoElement") as HTMLVideoElement;
  outputCanvas = document.getElementById("outputCanvas") as HTMLCanvasElement;
  outputCtx = outputCanvas.getContext("2d")!;
  
  drawingCanvas = document.createElement("canvas");
  drawingCtx = drawingCanvas.getContext("2d")!;

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.7,
    minHandPresenceConfidence: 0.7,
    minTrackingConfidence: 0.7
  });

  navigator.mediaDevices.getUserMedia({ video: { width: 1920, height: 1080 } }).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}


let lastVideoTime = -1;
function predictWebcam() {
  if (drawingCanvas.width !== video.videoWidth) {
    drawingCanvas.width = video.videoWidth;
    drawingCanvas.height = video.videoHeight;
    outputCanvas.width = video.videoWidth;
    outputCanvas.height = video.videoHeight;
  }

  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    const results = handLandmarker.detectForVideo(video, startTimeMs);

    processResults(results);
  }

  render();
  requestAnimationFrame(predictWebcam);
}

function getDistance(lm1: any, lm2: any) {
  return Math.hypot(lm1.x - lm2.x, lm1.y - lm2.y, (lm1.z || 0) - (lm2.z || 0));
}

function processResults(results: any) {
  let drawMode = false;
  let clearMode = false;
  let pauseMode = false;

  let currentDrawColor = drawColor;
  if (drawColor === 'rainbow') {
    rainbow_hue = (rainbow_hue + 1) % 360;
    currentDrawColor = `hsl(${rainbow_hue}, 100%, 50%)`;
  } else if (drawColor === 'eraser') {
    currentDrawColor = 'rgba(0,0,0,1)';
  }

  if (results.landmarks && results.landmarks.length > 0) {
    const lm = results.landmarks[0];
    const w = drawingCanvas.width;
    const h = drawingCanvas.height;

    // MediaPipe web outputs landmarks.x, landmarks.y in normalized coordinates
    // When the output canvas is rendered, it is flipped horizontally via scale(-1, 1).
    // Therefore, we MUST NOT flip the coordinates here, otherwise they will be flipped twice
    // and draw on the opposite side of the screen.
    const ix = lm[8].x * w;
    const iy = lm[8].y * h;

    if (smooth_x === null || smooth_y === null) {
      smooth_x = ix;
      smooth_y = iy;
    } else {
      const dx = ix - smooth_x;
      const dy = iy - smooth_y;
      const dist = Math.hypot(dx, dy);
      
      // Dynamic alpha: adaptive smoothing based on movement speed
      // Slow movement -> lower alpha -> more smoothing (reduces jitter)
      // Fast movement -> higher alpha -> less smoothing (reduces lag)
      // Adjusted for better small-detail drawing accuracy:
      let dynamicAlpha = 0.4 + (dist / 80);
      dynamicAlpha = Math.min(Math.max(dynamicAlpha, 0.4), 0.95);

      smooth_x = smooth_x + dynamicAlpha * dx;
      smooth_y = smooth_y + dynamicAlpha * dy;
    }

    // Compare Tip to PIP joint (6, 10, 14, 18). If curled, Tip is closer to wrist than PIP.
    const index_up = getDistance(lm[8], lm[0]) > getDistance(lm[6], lm[0]);
    const middle_up = getDistance(lm[12], lm[0]) > getDistance(lm[10], lm[0]);
    const ring_up = getDistance(lm[16], lm[0]) > getDistance(lm[14], lm[0]);
    const pinky_up = getDistance(lm[20], lm[0]) > getDistance(lm[18], lm[0]);
    const thumb_up = getDistance(lm[4], lm[0]) > getDistance(lm[3], lm[0]);
    
    const total_fingers = [index_up, middle_up, ring_up, pinky_up, thumb_up].filter(Boolean).length;

    if (total_fingers === 5) {
      clear_frames++;
      if (clear_frames > 15) {
        clearMode = true;
        clear_frames = 0;
      }
    } else {
      clear_frames = 0;
      if (index_up && middle_up && !ring_up && !pinky_up) {
        pauseMode = true;
      } else if (index_up) {
        drawMode = true;
      }
    }

    if (drawMode) {
      if (!isDrawing) {
        saveState();
        isDrawing = true;
      }

      drawingCtx.lineCap = "round";
      drawingCtx.lineJoin = "round";
      drawingCtx.lineWidth = drawThickness;

      if (drawColor === 'eraser') {
        drawingCtx.globalCompositeOperation = "destination-out";
        drawingCtx.strokeStyle = "rgba(0,0,0,1)";
      } else {
        drawingCtx.globalCompositeOperation = "source-over";
        drawingCtx.strokeStyle = currentDrawColor;
      }

      if (lineStyle === 'normal') {
        if (prev_x === null) {
          prev_x = smooth_x; prev_y = smooth_y;
          last_mid_x = smooth_x; last_mid_y = smooth_y;
        } else {
          const mid_x = (prev_x! + smooth_x!) / 2;
          const mid_y = (prev_y! + smooth_y!) / 2;
          
          drawingCtx.beginPath();
          drawingCtx.moveTo(last_mid_x!, last_mid_y!);
          drawingCtx.quadraticCurveTo(prev_x!, prev_y!, mid_x, mid_y);
          drawingCtx.stroke();
          
          prev_x = smooth_x; prev_y = smooth_y;
          last_mid_x = mid_x; last_mid_y = mid_y;
        }
      } else if (lineStyle === 'dotted') {
        if (prev_x === null) {
          prev_x = smooth_x; prev_y = smooth_y;
          drawingCtx.fillStyle = currentDrawColor;
          drawingCtx.beginPath();
          drawingCtx.arc(smooth_x!, smooth_y!, Math.max(2, drawThickness / 2), 0, Math.PI * 2);
          drawingCtx.fill();
        } else {
          const dist = Math.hypot(smooth_x! - prev_x!, smooth_y! - prev_y!);
          if (dist > drawThickness * 2.5) {
            drawingCtx.fillStyle = currentDrawColor;
            drawingCtx.beginPath();
            drawingCtx.arc(smooth_x!, smooth_y!, Math.max(2, drawThickness / 2), 0, Math.PI * 2);
            drawingCtx.fill();
            prev_x = smooth_x; prev_y = smooth_y;
          }
        }
      } else if (['straight', 'rectangle', 'circle'].includes(lineStyle)) {
        if (start_x === null) {
          start_x = smooth_x; start_y = smooth_y;
        }
        last_x = smooth_x; last_y = smooth_y;
      }
      lost_frames = 0;
    } else if (pauseMode) {
      isDrawing = false;
      prev_x = smooth_x; prev_y = smooth_y;
      start_x = smooth_x; start_y = smooth_y;
      last_x = smooth_x; last_y = smooth_y;
      lost_frames = 0;
    } else {
      isDrawing = false;
      lost_frames++;
      finalizeShape(currentDrawColor);
    }

    if (clearMode) {
      saveState();
      drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
      resetPoints();
    }
  } else {
    isDrawing = false;
    lost_frames++;
    let currentDrawColor = drawColor;
    if (drawColor === 'rainbow') currentDrawColor = `hsl(${rainbow_hue}, 100%, 50%)`;
    finalizeShape(currentDrawColor);
  }
}

function finalizeShape(currentDrawColor: string) {
  if (lost_frames > max_lost_frames) {
    if (start_x !== null && last_x !== null) {
      drawingCtx.lineCap = "round";
      drawingCtx.lineJoin = "round";
      drawingCtx.lineWidth = drawThickness;
      if (drawColor === 'eraser') {
        drawingCtx.globalCompositeOperation = "destination-out";
        drawingCtx.strokeStyle = "rgba(0,0,0,1)";
      } else {
        drawingCtx.globalCompositeOperation = "source-over";
        drawingCtx.strokeStyle = currentDrawColor;
      }

      drawingCtx.beginPath();
      if (lineStyle === 'straight') {
        drawingCtx.moveTo(start_x, start_y!);
        drawingCtx.lineTo(last_x, last_y!);
      } else if (lineStyle === 'rectangle') {
        drawingCtx.rect(start_x, start_y!, last_x - start_x, last_y! - start_y!);
      } else if (lineStyle === 'circle') {
        const radius = Math.hypot(last_x - start_x, last_y! - start_y!);
        drawingCtx.arc(start_x, start_y!, radius, 0, Math.PI * 2);
      }
      drawingCtx.stroke();
    }
    resetPoints();
  }
}

function resetPoints() {
  prev_x = null; prev_y = null;
  start_x = null; start_y = null;
  last_x = null; last_y = null;
  smooth_x = null; smooth_y = null;
  last_mid_x = null; last_mid_y = null;
}

function render() {
  const w = outputCanvas.width;
  const h = outputCanvas.height;
  outputCtx.clearRect(0, 0, w, h);

  if (bgMode === 'camera') {
    outputCtx.save();
    outputCtx.scale(-1, 1);
    outputCtx.drawImage(video, -w, 0, w, h);
    outputCtx.restore();
  } else if (bgMode === 'black') {
    outputCtx.fillStyle = 'black';
    outputCtx.fillRect(0, 0, w, h);
  } else if (bgMode === 'white') {
    outputCtx.fillStyle = 'white';
    outputCtx.fillRect(0, 0, w, h);
  }

  outputCtx.drawImage(drawingCanvas, 0, 0);

  // Draw temp shape
  if (start_x !== null && last_x !== null && lost_frames <= max_lost_frames && ['straight', 'rectangle', 'circle'].includes(lineStyle)) {
    outputCtx.lineCap = "round";
    outputCtx.lineJoin = "round";
    outputCtx.lineWidth = drawThickness;
    
    let currentDrawColor = drawColor;
    if (drawColor === 'rainbow') currentDrawColor = `hsl(${rainbow_hue}, 100%, 50%)`;
    if (drawColor === 'eraser') {
      outputCtx.strokeStyle = "rgba(0,0,0,1)";
      // Wait, eraser for shapes might look weird temporarily, but it's fine
    } else {
      outputCtx.strokeStyle = currentDrawColor;
    }

    outputCtx.beginPath();
    if (lineStyle === 'straight') {
      outputCtx.moveTo(start_x, start_y!);
      outputCtx.lineTo(last_x, last_y!);
    } else if (lineStyle === 'rectangle') {
      outputCtx.rect(start_x, start_y!, last_x - start_x, last_y! - start_y!);
    } else if (lineStyle === 'circle') {
      const radius = Math.hypot(last_x - start_x, last_y! - start_y!);
      outputCtx.arc(start_x, start_y!, radius, 0, Math.PI * 2);
    }
    outputCtx.stroke();
  }
}

function saveState() {
  if (drawingCanvas.width > 0 && drawingCanvas.height > 0) {
    undo_stack.push(drawingCtx.getImageData(0, 0, drawingCanvas.width, drawingCanvas.height));
    if (undo_stack.length > 20) undo_stack.shift();
    redo_stack = [];
  }
}

// Exposed API
(window as any).setColor = (colorName: string, btn: HTMLElement) => {
  if(btn && colorName !== 'eraser') {
    document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('colorPickerWrapper')?.classList.remove('active');
    btn.classList.add('active');
    document.querySelectorAll('.action-btn').forEach(b => b.classList.remove('active'));
  } else if (btn && colorName === 'eraser') {
    document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('colorPickerWrapper')?.classList.remove('active');
    btn.classList.add('active');
  }

  drawThickness = 6;
  if (colorName === 'red') drawColor = '#EF4444';
  else if (colorName === 'blue') drawColor = '#3B82F6';
  else if (colorName === 'green') drawColor = '#10B981';
  else if (colorName === 'yellow') drawColor = '#F59E0B';
  else if (colorName === 'purple') drawColor = '#8B5CF6';
  else if (colorName === 'pink') drawColor = '#EC4899';
  else if (colorName === 'cyan') drawColor = '#06B6D4';
  else if (colorName === 'orange') drawColor = '#F97316';
  else if (colorName === 'white') drawColor = '#FFFFFF';
  else if (colorName === 'rainbow') drawColor = 'rainbow';
  else if (colorName === 'eraser') {
    drawColor = 'eraser';
    drawThickness = 40;
  }
  else if (colorName.startsWith('hex_')) {
    drawColor = '#' + colorName.substring(4);
  }
};

(window as any).setCustomColor = (hexColor: string) => {
  document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.action-btn').forEach(b => b.classList.remove('active'));
  
  const wrapper = document.getElementById('colorPickerWrapper');
  if (wrapper) {
    wrapper.classList.add('active');
    wrapper.style.setProperty('--current-custom-color', hexColor);
  }
  drawColor = hexColor;
};

(window as any).clearCanvas = () => {
  saveState();
  drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
};

(window as any).undo = () => {
  if (undo_stack.length > 0) {
    redo_stack.push(drawingCtx.getImageData(0, 0, drawingCanvas.width, drawingCanvas.height));
    drawingCtx.putImageData(undo_stack.pop()!, 0, 0);
  }
};

(window as any).redo = () => {
  if (redo_stack.length > 0) {
    undo_stack.push(drawingCtx.getImageData(0, 0, drawingCanvas.width, drawingCanvas.height));
    drawingCtx.putImageData(redo_stack.pop()!, 0, 0);
  }
};

(window as any).setThickness = (val: string) => {
  drawThickness = parseInt(val, 10);
  const viewer = document.getElementById('brushSizeViewer');
  if (viewer) viewer.innerText = val + 'px';
};

(window as any).setBg = (mode: any, btn: HTMLElement) => {
  document.querySelectorAll('.bg-mode-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  bgMode = mode;
};

(window as any).setStyle = (mode: any, btn: HTMLElement) => {
  document.querySelectorAll('.style-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  lineStyle = mode;
  resetPoints();
};

(window as any).saveImage = () => {
  const link = document.createElement('a');
  link.download = 'auradraw_masterpiece.png';
  link.href = outputCanvas.toDataURL('image/png');
  link.click();
};

// Remove the old inline script implementations if they exist or just rely on these overriding them.
// Since these functions are placed on window, they will replace the inline ones if loaded after, or we can just ensure they execute.

init();
