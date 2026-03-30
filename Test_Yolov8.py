from inference import get_model
import supervision as sv
from inference.core.utils.image_utils import load_image_bgr
 
image = load_image_bgr("visual_recognition/step26/result.jpg")
model = get_model(model_id="yolov8n-640")
results = model.infer(image)[0]
results = sv.Detections.from_inference(results)
annotator = sv.BoxAnnotator(thickness=4)
annotated_image = annotator.annotate(image, results)
annotator = sv.LabelAnnotator(text_scale=2, text_thickness=2)
annotated_image = annotator.annotate(annotated_image, results)

# 将annotated_image保存成本地文件
# sv.save_image("annotated_image.png", annotated_image)   

sv.plot_image(annotated_image)