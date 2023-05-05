import { FaceMesh } from "@mediapipe/face_mesh";
import React, { useRef, useEffect } from "react";
import * as cam from "@mediapipe/camera_utils";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import * as ort from 'onnxruntime-web';

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  let camera = useRef(null);

  async function processImage(image) {
    const imageSize = 256;
    // dtype to float
    // image = image.cast("float32").div(255);
    // resize the image
    image = tf.image.resizeBilinear(image, [imageSize, imageSize]); // can also use tf.image.resizeNearestNeighbor
    const img_mean = tf.tensor([0.485, 0.456, 0.406], [1, 1, 3], "float32");
    const img_std = tf.tensor([0.229, 0.224, 0.225], [1, 1, 3], "float32");
    // Normalization
    const normalized = tf.div(tf.sub(tf.div(image, 255.0), img_mean), img_std);
    image = normalized.expandDims(); // to add the most left axis of size 1
    return image;
  }
  async function topKIndices(x, k) {
    if (!(x instanceof tf.Tensor)) {
        x = tf.tensor(x);
    }

    const result = await tf.topk(x.flatten(), k, true);
    const flatIndicesArray = await result.indices.array();

    const unraveledIndices = [];
    const shape = x.shape;
    for (let i = 0; i < flatIndicesArray.length; i++) {
        let index = flatIndicesArray[i];
        const unraveledIndex = [];
        for (let j = shape.length - 1; j >= 0; j--) {
            unraveledIndex[j] = index % shape[j];
            index = Math.floor(index / shape[j]);
        }
        unraveledIndices.push(unraveledIndex);
    }
    return unraveledIndices;
}

  async function getPeakLocation(heatmap, imageSize = [256, 256]) {
    const hHeight = 64;
    const hWidth = 64;
    const [[y1, y2], [x1, x2]] = await topKIndices(heatmap, 2);
    const x = ((x1 + (x2 - x1) / 4) / hWidth) * imageSize[0];
    const y = ((y1 + (y2 - y1) / 4) / hHeight) * imageSize[1];
    return [Math.floor(x), Math.floor(y)];
  }
  async function parseHeatmaps(heatmaps, imageSize) {
    const marks = [];
    heatmaps = tf.transpose(heatmaps, [2, 0, 1]);
    const numHeatmaps = heatmaps.shape[0];

    for (let i = 0; i < numHeatmaps; i++) {
      const heatmap = heatmaps.slice([i, 0, 0], [1, -1, -1]).squeeze();
      marks.push(await getPeakLocation(heatmap, imageSize));
    }
    return marks;
  }
  async function getEyeCoords(results) {
    // let X1 = null,
    //   X2 = null,
    //   Y1 = null,
    //   Y2 = null;
    // const model = await tf.loadGraphModel("/tfjs_graph_model/model.json");
    const session = await ort.InferenceSession.create('./model.onnx');
    // const canvas = canvasRef.current;
    // const ctx = canvas.getContext("2d");
    // if (results.multiFaceLandmarks) {
    //   for (const landmarks of results.multiFaceLandmarks) {
    //     var imgw = results.image.width;
    //     var imgh = results.image.height;
    //     X1 = Math.floor(landmarks[362].x * imgw - 10);
    //     Y1 = Math.floor(landmarks[386].y * imgh - 10);

    //     X2 = Math.floor(landmarks[263].x * imgw + 10);
    //     // Y2 =Math.floor(landmarks[374].y * imgh + 10);
    //   }
    //   ctx.clearRect(0, 0, canvas.width, canvas.height);
    //   var cropped = tf.slice(
    //     tf.browser.fromPixels(webcamRef.current.video),
    //     [Y1 - 10, X1, 0],
    //     [X2 - X1, X2 - X1, 3]
    //   );
    //   var newImg = await processImage(cropped);
    //   const output = model.predict(newImg);
    //   console.log(tf.tensor(output.arraySync()[0]).shape)
    //   // console.log(parseHeatmaps(tf.tensor(output.arraySync()[0]),(256,256)));
    // }
    var newImg = await processImage(results);
    // const output = model.predict(newImg);
    const output = await session.run(results);
    console.log(await output.array())
    // console.log(await parseHeatmaps(tf.tensor(output.arraySync()[0]), [256, 256]));
  }

  useEffect(() => {
    // const faceMesh = new FaceMesh({
    //   locateFile: (file) => {
    //     return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
    //   },
    // });

    // faceMesh.setOptions({
    //   maxNumFaces: 1,
    //   minDetectionConfidence: 0.5,
    //   minTrackingConfidence: 0.5,
    // });

    // faceMesh.onResults(getEyeCoords);
    // if (
    //   typeof webcamRef.current !== "undefined" &&
    //   webcamRef.current !== null
    // ) {
    //   camera.current = new cam.Camera(webcamRef.current.video, {
    //     onFrame: async () => {
    //       await faceMesh.send({ image: webcamRef.current.video });
    //     },
    //     width: 640,
    //     height: 480,
    //   });
    //   camera.current.start();
    // }
    const img = new Image();
    img.src = "186.jpg";
    const tensor = tf.browser.fromPixels(img);
    getEyeCoords(tensor);
  }, []);
  return (
    <div className="App">
      {/* <Webcam audio={false} ref={webcamRef} /> */}
      <canvas ref={canvasRef} />
    </div>
  );
}

export default App;
