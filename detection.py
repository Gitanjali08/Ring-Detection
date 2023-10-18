{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "22467f06",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\shind\\anaconda3\\lib\\site-packages\\tensorflow\\python\\autograph\\pyct\\static_analysis\\liveness.py:83: Analyzer.lamba_check (from tensorflow.python.autograph.pyct.static_analysis.liveness) is deprecated and will be removed after 2023-09-23.\n",
      "Instructions for updating:\n",
      "Lambda fuctions will be no more assumed to be used in the statement where they are used, or at least in the same block. https://github.com/tensorflow/tensorflow/issues/56089\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "from object_detection.utils import label_map_util\n",
    "from object_detection.utils import visualization_utils as viz_utils\n",
    "from object_detection.builders import model_builder\n",
    "import cv2 \n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from object_detection.utils import config_util\n",
    "from object_detection.protos import pipeline_pb2\n",
    "from google.protobuf import text_format\n",
    "# Load pipeline config and build a detection model\n",
    "configs = config_util.get_configs_from_pipeline_file('Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config')\n",
    "detection_model = model_builder.build(model_config=configs['model'], is_training=False)\n",
    "\n",
    "# Restore checkpoint\n",
    "ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)\n",
    "ckpt.restore(os.path.join('Tensorflow/workspace/models/my_ssd_mobnet/', 'ckpt-6')).expect_partial()\n",
    "@tf.function\n",
    "def detect_fn(image):\n",
    "    image, shapes = detection_model.preprocess(image)\n",
    "    prediction_dict = detection_model.predict(image, shapes)\n",
    "    detections = detection_model.postprocess(prediction_dict, shapes)\n",
    "    return detections\n",
    "category_index = label_map_util.create_category_index_from_labelmap('Tensorflow/workspace/annotations/label_map.pbtxt')\n",
    "cap = cv2.VideoCapture(0)\n",
    "width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))\n",
    "height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))\n",
    "while True: \n",
    "    ret, frame = cap.read()\n",
    "    image_np = np.array(frame)\n",
    "    \n",
    "    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)\n",
    "    detections = detect_fn(input_tensor)\n",
    "    \n",
    "    num_detections = int(detections.pop('num_detections'))\n",
    "    detections = {key: value[0, :num_detections].numpy()\n",
    "                  for key, value in detections.items()}\n",
    "    detections['num_detections'] = num_detections\n",
    "\n",
    "    # detection_classes should be ints.\n",
    "    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)\n",
    "\n",
    "    label_id_offset = 1\n",
    "    image_np_with_detections = image_np.copy()\n",
    "\n",
    "    viz_utils.visualize_boxes_and_labels_on_image_array(\n",
    "                image_np_with_detections,\n",
    "                detections['detection_boxes'],\n",
    "                detections['detection_classes']+label_id_offset,\n",
    "                detections['detection_scores'],\n",
    "                category_index,\n",
    "                use_normalized_coordinates=True,\n",
    "                max_boxes_to_draw=5,\n",
    "                min_score_thresh=.5,\n",
    "                agnostic_mode=False)\n",
    "\n",
    "    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))\n",
    "    \n",
    "    if cv2.waitKey(1) & 0xFF == ord('q'):\n",
    "        cap.release()\n",
    "        break"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "589ffc1b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
