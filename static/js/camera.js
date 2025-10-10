const video = document.getElementById("webcam");
const canvas = document.getElementById("outputCanvas");
const ctx = canvas.getContext("2d");
const sendBtn = document.getElementById("sendBtn");
const sendCanvas = document.createElement("canvas");
const sendCtx = sendCanvas.getContext("2d");

let lastHand = null;

// 🔹 手の骨格の接続定義
const fingers = [
    [0, 1, 2, 3, 4],      // 親指
    [0, 5, 6, 7, 8],      // 人差し指
    [0, 9, 10, 11, 12],   // 中指
    [0, 13, 14, 15, 16],  // 薬指
    [0, 17, 18, 19, 20],  // 小指
];

let detector;
const videoWidth = 960;
const videoHeight = 680;

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: videoWidth, height: videoHeight }
    });
    video.srcObject = stream;
    await video.play();

    // video の実際の解像度に canvas を揃える
    video.width = video.videoWidth;
    video.height = video.videoHeight;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
}

async function setupModel() {
    const model = handPoseDetection.SupportedModels.MediaPipeHands;
    detector = await handPoseDetection.createDetector(model, {
        runtime: "mediapipe",
        modelType: "full",
        maxHands: 1,
        solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/hands",
    });
}

function drawSkeleton(keypoints, scaleX = 1, scaleY = 1) {
    ctx.strokeStyle = "cyan";
    ctx.lineWidth = 2;

    // 指の線を描く
    fingers.forEach((finger) => {
        ctx.beginPath();
        finger.forEach((idx, i) => {
            const pt = keypoints[idx];
            const x = pt.x * scaleX;
            const y = pt.y * scaleY;
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();
    });

    // 関節点（赤丸）
    keypoints.forEach((pt) => {
        ctx.beginPath();
        ctx.arc(pt.x * scaleX, pt.y * scaleY, 4, 0, 2 * Math.PI);
        ctx.fillStyle = "red";
        ctx.fill();
    });
}

async function detect() {
    // スケーリングを毎フレームリセット
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const hands = await detector.estimateHands(video);

    if (hands.length > 0) {
        lastHand = hands[0].keypoints;

        // 表示サイズに合わせてスケール補正
        const scaleX = canvas.clientWidth / canvas.width;
        const scaleY = canvas.clientHeight / canvas.height;

        drawSkeleton(lastHand, scaleX, scaleY);

        sendBtn.disabled = false;
        sendBtn.classList.add("bg-blue-500");
    } else {
        lastHand = null;
        sendBtn.disabled = true;
        sendBtn.classList.remove("bg-blue-500");
    }

    requestAnimationFrame(detect);
}

// 🔘 切り抜いて送信
function sendImage() {
    if (!lastHand) return;

    const xs = lastHand.map(pt => pt.x);
    const ys = lastHand.map(pt => pt.y);

    // 基本マージン
    const margin = 100;
    // 手首側（下方向）だけ追加マージン
    const extraWristMargin = 200;

    let minX = Math.max(Math.min(...xs) - margin, 0);
    let minY = Math.max(Math.min(...ys) - margin, 0);
    let maxX = Math.min(Math.max(...xs) + margin, video.videoWidth);
    let maxY = Math.min(Math.max(...ys) + margin, video.videoHeight);

    let cropWidth = maxX - minX;
    let cropHeight = maxY - minY;

    // 🟦 上下に余白をつける
    const extraPadding = 100; // ← 上下に追加する黒余白のピクセル数
    sendCanvas.width = cropWidth;
    sendCanvas.height = cropHeight + extraPadding * 2;

    // 背景を白で塗りつぶす
    sendCtx.fillStyle = "white";
    sendCtx.fillRect(0, 0, sendCanvas.width, sendCanvas.height);

    // 手の切り抜きを中央（上下に余白を残して）貼り付ける
    sendCtx.drawImage(
        video,
        minX, minY, cropWidth, cropHeight, // 元映像から切り抜く範囲
        0, extraPadding, cropWidth, cropHeight // キャンバスに描画位置（上下余白を確保）
    );

    sendCanvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append("file", blob, "hand.jpg");

        // FastAPI に送信
        const res = await fetch("/predict", { method: "POST", body: formData });
        const data = await res.json();
        console.log(data);

        if (res.ok) {
            document.getElementById("res-class").textContent = data.predicted_class;
            const imgEl = document.getElementById("res-image");
            imgEl.src = `${data.image_url}?t=${Date.now()}`;
            imgEl.classList.remove("hidden");
        }
    }, "image/jpeg");
}

sendBtn.addEventListener("click", sendImage);
document.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        sendImage();
    }
});

async function main() {
    await setupCamera();
    await setupModel();
    detect();
}

main();
