import { py, python, PyClass } from "pythonia";
const tf = await python("tensorflow");

class KerasCallback extends PyClass {
  constructor() {
    super(tf.keras.callbacks.Callback);
  }

  on_epoch_end(epock, logs) {
    if (logs.loss < 0.4) {
      console.log("/nReached 60% accuracy so cancelling training");
      this.model.stop_training;
    }
  }
}

const mnist = await tf.keras.datasets.fashion_mnist;
const [[training_images, training_labels], [test_images, test_labels]] =
  await mnist.load_data;

const trainingImages = await py`${training_images}`;
