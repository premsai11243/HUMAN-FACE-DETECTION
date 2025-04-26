
<h1>Human Face Detection with YOLOv8</h1>

<h2> Project Overview</h2>
<p>
A complete pipeline for detecting human faces in images and videos using YOLOv8, with preprocessing, augmentation, training, evaluation, and deployment.
</p>

<hr>

<h2> Steps</h2>

<ol>
  <li><strong>Import Libraries</strong><br>
    Pandas, NumPy, OpenCV, PIL, OS, PyTorch.
  </li>

  <li><strong>Data Preprocessing</strong><br>
    - Clean annotations (CSV).<br>
    - Resize images for efficient GPU training.<br>
    - Integrate images and labels.
  </li>

  <li><strong>Data Augmentation</strong><br>
    - Apply flips, brightness, contrast, rotation, and normalization.
  </li>

  <li><strong>Exploratory Data Analysis</strong><br>
    - Visualize images with bounding boxes.<br>
    - Count images and faces.
  </li>

  <li><strong>Training Preparation</strong><br>
    - Convert CSV to YOLO TXT format.<br>
    - Create dataset splits (train/val).<br>
    - Define YAML configuration.
  </li>

  <li><strong>Model Training</strong><br>
    - Train YOLOv8 with custom data.<br>
    - Monitor metrics (Loss, mAP, Precision, Recall).
  </li>

  <li><strong>Model Evaluation</strong><br>
    - Evaluate on images, videos, and webcam streams.
  </li>

  <li><strong>Deployment</strong><br>
    - Streamlit app for real-time predictions.<br>
    - Project report documenting methodology, analysis, and results.
  </li>
</ol>

<hr>

<h2> Final Goal</h2>
<p>
Accurate and fast human face detection system ready for real-world applications!
</p>
