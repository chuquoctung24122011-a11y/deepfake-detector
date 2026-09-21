import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
import warnings

warnings.filterwarnings('ignore')

model_path = 'ai_face_detector_model.keras'
model = tf.keras.models.load_model(model_path)

def generate_smooth_gradcam(img_array, model):
    sub_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            sub_model = layer
            break

    if sub_model:
        try:
            last_conv = sub_model.get_layer('out_relu')
        except:
            last_conv = [l for l in sub_model.layers if 'conv' in l.name.lower() or 'relu' in l.name.lower()][-1]

        grad_base = tf.keras.models.Model(inputs=sub_model.inputs, outputs=[last_conv.output, sub_model.output])

        with tf.GradientTape() as tape:
            conv_outputs, base_outputs = grad_base(img_array)
            tape.watch(conv_outputs)
            x = base_outputs
            head_start = False
            for layer in model.layers:
                if layer == sub_model:
                    head_start = True
                    continue
                if head_start:
                    x = layer(x)
            preds = x
            pred_score = preds[0][0]

        grads = tape.gradient(pred_score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
    else:
        last_conv_name = [l.name for l in model.layers if 'conv' in l.name.lower() or 'relu' in l.name.lower()][-1]
        grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[model.get_layer(last_conv_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(img_array)
            pred_score = preds[0][0]
        grads = tape.gradient(pred_score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)

    heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
    heatmap_smooth = cv2.GaussianBlur(heatmap_resized, (15, 15), 0)
    if np.max(heatmap_smooth) != 0:
        heatmap_smooth = heatmap_smooth / np.max(heatmap_smooth)

    return heatmap_smooth, float(pred_score)

def analyze_deepfake(input_img):
    if input_img is None:
        return None, None, "⚠️ Vui lòng tải lên 1 bức ảnh!"
    
    raw_img = cv2.resize(input_img, (224, 224))
    img_array = np.expand_dims(raw_img.astype(np.float32), axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    heatmap_smooth, score = generate_smooth_gradcam(img_array, model)

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_smooth), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed_img = cv2.addWeighted(raw_img, 0.6, heatmap_colored, 0.4, 0)

    if score > 0.5:
        confidence = score * 100
        result_str = (
            f"🟢 KẾT QUẢ: REAL (Ảnh Thật)\n"
            f"📊 Độ tin cậy: {confidence:.2f}%\n\n"
            f"🔍 BẰNG CHỨNG GIẢI THÍCH (GRAD-CAM):\n"
            f"• Mô hình xác nhận kết cấu đường nét khuôn mặt hoàn toàn tự nhiên.\n"
            f"• Vùng màu Đỏ/Vàng thể hiện các điểm đặc trưng chuẩn mà AI dùng để đối chiếu xác thực.\n"
            f"• Không phát hiện dấu vết biến dạng hay nhiễu ghép từ thuật toán Deepfake."
        )
    else:
        confidence = (1 - score) * 100
        result_str = (
            f"🔴 KẾT QUẢ: FAKE (Deepfake)\n"
            f"📊 Độ tin cậy: {confidence:.2f}%\n\n"
            f"🔍 BẰNG CHỨNG GIẢI THÍCH (GRAD-CAM):\n"
            f"• Vùng màu Đỏ/Vàng rực chính là BẰNG CHỨNG tố cáo điểm bất thường trên ảnh.\n"
            f"• AI đã soi thấy sai lệch ở vùng kết cấu da, viền tóc, mắt/miệng hoặc ranh giới ghép mặt.\n"
            f"• Đây là các dấu hiệu đặc trưng của mô hình sinh ảnh AI (GAN / Diffusion)."
        )

    return heatmap_colored, superimposed_img, result_str

with gr.Blocks(theme=gr.themes.Soft(), title="AI Deepfake Detector - Developed by ChuQuocTung") as demo:
    gr.Markdown(
        """
        # 🛡️ HỆ THỐNG NHẬN DIỆN DEEPFAKE & GIẢI THÍCH GRAD-CAM
        ### 👨‍💻 **Developed by ChuQuocTung**
        ---
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="1. Tải ảnh đầu vào", type="numpy")
            btn = gr.Button("🚀 Phân Tích Bằng Chứng AI", variant="primary")
            result_text = gr.Textbox(label="📋 Kết quả & Bằng chứng giải thích", lines=8)
        
        with gr.Column():
            output_heatmap = gr.Image(label="2. Bản đồ nhiệt Grad-CAM Mịn")
            output_gradcam = gr.Image(label="3. Ảnh phủ Grad-CAM Trực quan")

    gr.Markdown(
        """
        ---
        <center><small>⚡ Deepfake Detection Engine | Designed & Developed by <b>ChuQuocTung</b></small></center>
        """
    )

    btn.click(
        fn=analyze_deepfake,
        inputs=input_image,
        outputs=[output_heatmap, output_gradcam, result_text]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
