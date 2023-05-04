import { FaceMesh } from "@mediapipe/face_mesh";
import React, { useRef, useEffect } from "react";
import * as cam from "@mediapipe/camera_utils";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  let camera = useRef(null);

  async function processImage(image) {
    const imageSize = 256;
    // dtype to float
    image = image.cast("float32").div(255);
    // resize the image
    image = tf.image.resizeBilinear(image, ([imageSize, imageSize])); // can also use tf.image.resizeNearestNeighbor
    image = image.expandDims(); // to add the most left axis of size 1
    return image;
  }
  async function getEyeCoords(results) {
    let X1 = null,
      X2 = null,
      Y1 = null,
      Y2 = null;
    const model = await tf.loadGraphModel("/tfjs_graph_model/model.json");
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (results.multiFaceLandmarks) {
      for (const landmarks of results.multiFaceLandmarks) {
        var imgw = results.image.width;
        var imgh = results.image.height;
        X1 = Math.floor(landmarks[362].x * imgw - 10);
        Y1 = Math.floor(landmarks[386].y * imgh - 10);

        X2 = Math.floor(landmarks[263].x * imgw + 10);
        // Y2 =Math.floor(landmarks[374].y * imgh + 10);
      }
      // var startingPoint = [Y1, X1, 0];
      // var newSize = [Y2 - Y1, X2 - X1, 3];
      // Lấy bức ảnh từ webcam và chuyển đổi thành Tensor
      // var image = tf.browser.fromPixels(webcamRef.current.video);
      //
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      var cropped = tf.slice(
        tf.browser.fromPixels(webcamRef.current.video),
        [Y1 - 10, X1, 0],
        [X2 - X1, X2 - X1, 3]
      );
      var newImg = await processImage(cropped)
      const output = model.predict(newImg);
      console.log(output)
      // var drawCanvas = async (pixels) => {
      //   var imageData = ctx.createImageData(
      //     cropped.shape[1],
      //     cropped.shape[0]
      //   );
      //   imageData.data.set(pixels);
      //   ctx.putImageData(imageData, 0, 0);
      // };
      // tf.browser.toPixels(cropped).then(drawCanvas);
    }
  }

  useEffect(() => {
    const faceMesh = new FaceMesh({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
      },
    });

    faceMesh.setOptions({
      maxNumFaces: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    faceMesh.onResults(getEyeCoords);
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null
    ) {
      camera.current = new cam.Camera(webcamRef.current.video, {
        onFrame: async () => {
          await faceMesh.send({ image: webcamRef.current.video });
        },
        width: 640,
        height: 480,
      });
      camera.current.start();
    }
  }, []);
  return (
    <div className="App">
      <Webcam audio={false} ref={webcamRef} />
      <canvas ref={canvasRef} />
    </div>
  );
}

export default App;
