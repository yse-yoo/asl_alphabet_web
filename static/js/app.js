const video = document.getElementById("webcam");
const canvas = document.getElementById("outputCanvas");
const ctx = canvas.getContext("2d");
const sendBtn = document.getElementById("sendBtn");

let detectorHands, detectorPose;
const videoWidth = 960;
const videoHeight = 680;

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: videoWidth, height: videoHeight },
  });
  video.srcObject = stream;
  await video.play();

  video.width = video.videoWidth;
  video.height = video.videoHeight;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

async function setupModels() {
  // --- Hands モデル ---
  const modelHands = handPoseDetection.SupportedModels.MediaPipeHands;
  detectorHands = await handPoseDetection.createDetector(modelHands, {
    runtime: "mediapipe",
    modelType: "full",
    maxHands: 2,
    solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/hands",
  });

  // --- Pose モデル ---
  const modelPose = poseDetection.SupportedModels.MoveNet;
  detectorPose = await poseDetection.createDetector(modelPose, {
    modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING, 
  });
}


async function detect() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const hands = await detectorHands.estimateHands(video);
  const poses = await detectorPose.estimatePoses(video);

  // 骨格を描画
  ctx.strokeStyle = "cyan";
  ctx.lineWidth = 2;

  // --- Pose 描画 ---
  if (poses.length > 0) {
    const keypoints = poses[0].keypoints;
    keypoints.forEach((pt) => {
      if (pt.score > 0.4) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = "lime";
        ctx.fill();
      }
    });
  }

  // --- Hands 描画 ---
  if (hands.length > 0) {
    hands.forEach((hand) => {
      hand.keypoints.forEach((pt) => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 3, 0, 2 * Math.PI);
        ctx.fillStyle = "red";
        ctx.fill();
      });
    });
    sendBtn.disabled = false;
    sendBtn.classList.add("bg-blue-500");
  } else {
    sendBtn.disabled = true;
    sendBtn.classList.remove("bg-blue-500");
  }

  requestAnimationFrame(detect);
}

// 🔘 上半身ごと送信
function sendUpperBody() {
  const sendCanvas = document.createElement("canvas");
  const sendCtx = sendCanvas.getContext("2d");

  // 全体のうち上半身（顔～胸あたり）を切り抜き
  const cropY = video.videoHeight * 0.15; // 顔より少し下
  const cropH = video.videoHeight * 0.55; // 胸のあたりまで
  const cropX = 0;
  const cropW = video.videoWidth;

  sendCanvas.width = cropW;
  sendCanvas.height = cropH;

  sendCtx.drawImage(
    video,
    cropX, cropY, cropW, cropH, // 元画像から上半身領域を切り抜き
    0, 0, cropW, cropH
  );

  sendCanvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append("file", blob, "upperbody.jpg");

    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (res.ok) {
      document.getElementById("res-class").textContent = data.predicted_class;
      const imgEl = document.getElementById("res-image");
      imgEl.src = `${data.image_url}?t=${Date.now()}`;
      imgEl.classList.remove("hidden");
    }
  }, "image/jpeg");
}

sendBtn.addEventListener("click", sendUpperBody);
document.addEventListener("keydown", (event) => {
  if (event.key === "Enter") sendUpperBody();
});

async function main() {
  await setupCamera();
  await setupModels();
  detect();
}

main();