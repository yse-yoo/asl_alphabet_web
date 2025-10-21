import * as handPoseDetection from "https://cdn.jsdelivr.net/npm/@tensorflow-models/hand-pose-detection";
import * as poseDetection from "https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection";

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
  // MediaPipe Hands
  const modelHands = handPoseDetection.SupportedModels.MediaPipeHands;
  detectorHands = await handPoseDetection.createDetector(modelHands, {
    runtime: "mediapipe",
    modelType: "full",
    maxHands: 2,
    solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/hands",
  });

  // Pose (MoveNet Lightning)
  const modelPose = poseDetection.SupportedModels.MoveNet;
  detectorPose = await poseDetection.createDetector(modelPose, {
    modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
  });
}

async function detect() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const hands = await detectorHands.estimateHands(video);
  const poses = await detectorPose.estimatePoses(video);

  // --- 骨格描画（任意） ---
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

  if (hands.length > 0) {
    hands.forEach((hand) => {
      hand.keypoints.forEach((pt) => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 3, 0, 2 * Math.PI);
        ctx.fillStyle = "red";
        ctx.fill();
      });
    });
  }

  requestAnimationFrame(detect);
}

async function sendLandmarks() {
  const hands = await detectorHands.estimateHands(video);
  const poses = await detectorPose.estimatePoses(video);

  const handData = hands.map((h) =>
    h.keypoints.map((pt) => ({ x: pt.x / video.videoWidth, y: pt.y / video.videoHeight, z: pt.z || 0 }))
  );

  const poseData =
    poses.length > 0
      ? poses[0].keypoints.map((pt) => ({
          x: pt.x / video.videoWidth,
          y: pt.y / video.videoHeight,
          z: 0,
        }))
      : [];

  const payload = { hands: handData.flat(), pose: poseData };

  const res = await fetch("/predict_json", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();

  document.getElementById("res-class").textContent = `${data.predicted_class} (${data.confidence})`;
}

sendBtn.addEventListener("click", sendLandmarks);

async function main() {
  await setupCamera();
  await setupModels();
  detect();
}

main();