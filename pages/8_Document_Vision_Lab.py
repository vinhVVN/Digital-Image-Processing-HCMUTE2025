import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import sys
from pathlib import Path

# Fix relative imports
sys.path.append(str(Path(__file__).parent.parent))

from doc_vision_lab.ocr_adapters.paddle_adapter import PaddleAdapter
from doc_vision_lab.metrics import calculate_cer, calculate_wer
from doc_vision_lab.quality_analyzer import ImageAnalyzer
from doc_vision_lab.hypothesis_generator import HypothesisGenerator
from doc_vision_lab.pipeline_engine import PipelineEngine
from doc_vision_lab.experiment import ExperimentTracker
import json

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "image_original" not in st.session_state:
    st.session_state.image_original = None
if "gt_text" not in st.session_state:
    st.session_state.gt_text = ""
if "baseline_metrics" not in st.session_state:
    st.session_state.baseline_metrics = None
if "image_metrics" not in st.session_state:
    st.session_state.image_metrics = None
if "current_pipeline" not in st.session_state:
    st.session_state.current_pipeline = []
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "experiment_metrics" not in st.session_state:
    st.session_state.experiment_metrics = None
if "tracker" not in st.session_state:
    st.session_state.tracker = ExperimentTracker()

# Cache the OCR engine to avoid reloading
@st.cache_resource
def get_paddle_engine():
    adapter = PaddleAdapter()
    adapter.load_model()
    return adapter

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("🔬 Settings")
st.sidebar.markdown("Upload a document image to begin.")

uploaded_file = st.sidebar.file_uploader("1. Upload Image", type=["jpg", "jpeg", "png", "bmp"])

enable_eval = st.sidebar.checkbox("☑️ Enable Evaluation Mode", value=True, help="Turn on to calculate CER/WER using Ground Truth")

if enable_eval:
    gt_input = st.sidebar.text_area("2. Ground Truth Text (Required for CER/WER)", height=150)
else:
    gt_input = ""

if uploaded_file is not None:
    # Read image into session state
    image = Image.open(uploaded_file)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Analyze only when new image is uploaded
    if st.session_state.image_original is None or not np.array_equal(st.session_state.image_original, img_bgr):
        st.session_state.image_original = img_bgr
        st.session_state.baseline_metrics = None 
        st.session_state.experiment_metrics = None
        st.session_state.processed_image = img_bgr # Default processed image is original
        analyzer = ImageAnalyzer()
        st.session_state.image_metrics = analyzer.analyze(img_bgr)

if gt_input:
    st.session_state.gt_text = gt_input

# ==========================================
# MAIN AREA
# ==========================================
st.title("Document Vision Lab 🔬")
st.markdown("*A Computer Vision Research Workspace for Document Understanding*")

# Tabs
tab1, tab2 = st.tabs(["🔬 The Laboratory", "📚 Experiment History"])

with tab1:
    if st.session_state.image_original is None:
        st.info("👈 Please upload an image from the sidebar to begin.")
    else:
        st.header("1. Ingestion & Baseline")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            # Convert BGR to RGB for display
            img_rgb = cv2.cvtColor(st.session_state.image_original, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)
            
        with col2:
            st.subheader("Baseline Metrics")
            if enable_eval and not st.session_state.gt_text:
                st.warning("⚠️ Please provide Ground Truth Text in the sidebar to calculate Error Rates.")
            
            if st.button("🚀 Establish Baseline (Run OCR)"):
                if enable_eval and not st.session_state.gt_text:
                    st.error("Ground Truth text is required for Evaluation Mode.")
                else:
                    with st.spinner("Running PaddleOCR..."):
                        try:
                            # 1. Get Engine
                            engine = get_paddle_engine()
                            
                            # 2. Extract Text
                            start_time = time.time()
                            extracted_text, confidence = engine.extract_text(st.session_state.image_original)
                            latency = time.time() - start_time
                            
                            # 3. Calculate Metrics
                            metrics_data = {
                                "latency": latency,
                                "confidence": confidence,
                                "extracted_text": extracted_text,
                                "mode": "eval" if enable_eval else "standard"
                            }
                            
                            if enable_eval:
                                metrics_data["cer"] = calculate_cer(st.session_state.gt_text, extracted_text)
                                metrics_data["wer"] = calculate_wer(st.session_state.gt_text, extracted_text)
                            
                            # 4. Save to Session State
                            st.session_state.baseline_metrics = metrics_data
                            st.success("Baseline established!")
                        except Exception as e:
                            st.error(f"OCR Error: {e}")
            
            # Display Baseline Metrics if they exist
            if st.session_state.baseline_metrics:
                m = st.session_state.baseline_metrics
                
                if m.get("mode") == "eval":
                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    mcol1.metric("CER", f"{m['cer']:.2f}%")
                    mcol2.metric("WER", f"{m['wer']:.2f}%")
                    mcol3.metric("Latency", f"{m['latency']:.2f}s")
                    mcol4.metric("Confidence", f"{m['confidence']:.2f}")
                else:
                    mcol1, mcol2 = st.columns(2)
                    mcol1.metric("Speed (Latency)", f"{m['latency']:.2f}s")
                    mcol2.metric("Confidence", f"{m['confidence']:.2f}")
                
                with st.expander("View Extracted Text"):
                    st.text(m['extracted_text'])
                    
        st.divider()
        st.header("2. Analysis & Hypothesis 💡")
        
        if st.session_state.image_metrics:
            metrics = st.session_state.image_metrics
            
            # Show Metrics
            qcol1, qcol2, qcol3 = st.columns(3)
            qcol1.metric("Blur Score (Laplacian Var)", f"{metrics['blur_score']:.1f}", help="Higher is sharper")
            qcol2.metric("Contrast Score (RMS)", f"{metrics['contrast_score']:.1f}", help="Higher is more contrast")
            qcol3.metric("Noise Estimate", f"{metrics['noise_score']:.2f}", help="Lower is less noise")
            
            # Show Hypotheses
            generator = HypothesisGenerator()
            hypotheses = generator.generate(metrics)
            
            st.info("📌 **Nhận định từ Hệ thống (Hypotheses):**")
            for hyp in hypotheses:
                st.markdown(f"- {hyp}")
                
        st.divider()
        st.header("3. Pipeline Builder ⚙️")
        st.markdown("Xây dựng chuỗi tiền xử lý ảnh để kiểm chứng giả thuyết.")
        
        pcol1, pcol2 = st.columns([1, 2])
        
        with pcol1:
            st.subheader("Add Step")
            step_name = st.selectbox("Choose operation:", list(PipelineEngine.AVAILABLE_STEPS.keys()))
            if st.button("➕ Add to Pipeline"):
                st.session_state.current_pipeline.append({
                    "step": step_name,
                    "params": PipelineEngine.AVAILABLE_STEPS[step_name]
                })
            
            if st.button("🗑️ Clear Pipeline"):
                st.session_state.current_pipeline = []
                st.session_state.processed_image = st.session_state.image_original
                st.session_state.experiment_metrics = None
                
            st.markdown("---")
            st.write("**Current Pipeline:**")
            if not st.session_state.current_pipeline:
                st.write("*(Empty)*")
            else:
                for i, step in enumerate(st.session_state.current_pipeline):
                    scol1, scol2 = st.columns([4, 1])
                    scol1.write(f"{i+1}. {step['step']}")
                    if scol2.button("❌", key=f"del_step_{i}", help="Delete this step"):
                        st.session_state.current_pipeline.pop(i)
                        st.rerun()
                        
            # Update processed image if pipeline changes
            if st.session_state.image_original is not None:
                engine = PipelineEngine()
                engine.load_pipeline(st.session_state.current_pipeline)
                st.session_state.processed_image = engine.execute(st.session_state.image_original)

        with pcol2:
            st.subheader("Image Comparison")
            if st.session_state.processed_image is not None:
                img_rgb_orig = cv2.cvtColor(st.session_state.image_original, cv2.COLOR_BGR2RGB)
                
                # Handle single channel (Grayscale/Binary) image display
                disp_img = st.session_state.processed_image
                if len(disp_img.shape) == 2:
                    disp_img = cv2.cvtColor(disp_img, cv2.COLOR_GRAY2RGB)
                else:
                    disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
                    
                icol1, icol2 = st.columns(2)
                with icol1:
                    st.image(img_rgb_orig, caption="Before (Original)", use_container_width=True)
                with icol2:
                    st.image(disp_img, caption="After (Processed)", use_container_width=True)

        st.divider()
        st.header("4. Experiment Execution 🚀")
        st.markdown("Chạy OCR trên ảnh đã xử lý và so sánh độ cải thiện.")
        
        if st.button("▶️ Run Experiment on Processed Image"):
            if st.session_state.baseline_metrics is None:
                st.error("⚠️ Bạn cần chạy Baseline trước khi thử nghiệm để có cơ sở đối chiếu!")
            else:
                with st.spinner("Running OCR on Processed Image..."):
                    try:
                        ocr_engine = get_paddle_engine()
                        start_time = time.time()
                        ext_text, conf = ocr_engine.extract_text(st.session_state.processed_image)
                        lat = time.time() - start_time
                        
                        exp_data = {
                            "latency": lat,
                            "confidence": conf,
                            "extracted_text": ext_text,
                            "mode": st.session_state.baseline_metrics.get("mode", "standard")
                        }
                        
                        if enable_eval:
                            exp_data["cer"] = calculate_cer(st.session_state.gt_text, ext_text)
                            exp_data["wer"] = calculate_wer(st.session_state.gt_text, ext_text)
                            
                        st.session_state.experiment_metrics = exp_data
                        st.success("Thí nghiệm hoàn tất!")
                    except Exception as e:
                        import traceback
                        st.error(f"Error running OCR:")
                        st.code(traceback.format_exc())
                        
        if st.session_state.experiment_metrics and st.session_state.baseline_metrics:
            em = st.session_state.experiment_metrics
            bm = st.session_state.baseline_metrics
            
            st.subheader("Delta Metrics (Sự thay đổi)")
            if enable_eval:
                dcol1, dcol2, dcol3, dcol4 = st.columns(4)
                cer_delta = em['cer'] - bm['cer']
                wer_delta = em['wer'] - bm['wer']
                lat_delta = em['latency'] - bm['latency']
                conf_delta = em['confidence'] - bm['confidence']
                
                dcol1.metric("CER", f"{em['cer']:.2f}%", f"{cer_delta:.2f}%", delta_color="inverse")
                dcol2.metric("WER", f"{em['wer']:.2f}%", f"{wer_delta:.2f}%", delta_color="inverse")
                dcol3.metric("Latency", f"{em['latency']:.2f}s", f"{lat_delta:.2f}s", delta_color="inverse")
                dcol4.metric("Confidence", f"{em['confidence']:.2f}", f"{conf_delta:.2f}", delta_color="normal")
            else:
                dcol1, dcol2 = st.columns(2)
                lat_delta = em['latency'] - bm['latency']
                conf_delta = em['confidence'] - bm['confidence']
                dcol1.metric("Speed (Latency)", f"{em['latency']:.2f}s", f"{lat_delta:.2f}s", delta_color="inverse")
                dcol2.metric("Confidence", f"{em['confidence']:.2f}", f"{conf_delta:.2f}", delta_color="normal")
                
            with st.expander("View Extracted Text"):
                st.text(em['extracted_text'])
                
            st.write("")
            if st.button("💾 Save to Workspace", type="primary"):
                st.session_state.tracker.add_experiment(
                    st.session_state.current_pipeline,
                    st.session_state.baseline_metrics,
                    st.session_state.experiment_metrics
                )
                st.success("Đã lưu kết quả vào Experiment History!")

with tab2:
    st.header("Experiment History 📚")
    st.markdown("Bảng tổng hợp kết quả các lần thí nghiệm so với Baseline.")
    
    df = st.session_state.tracker.get_history_df()
    st.dataframe(df, use_container_width=True)
    
    st.divider()
    st.subheader("Recipe Management 🛠️")
    st.markdown("Xuất chuỗi tiền xử lý (Pipeline) thành file JSON để chia sẻ hoặc nạp lại (Import) một Recipe có sẵn.")
    
    rcol1, rcol2 = st.columns(2)
    
    with rcol1:
        st.markdown("**Export Current Pipeline**")
        if st.session_state.current_pipeline:
            recipe_json = json.dumps(st.session_state.current_pipeline, indent=4)
            st.download_button(
                label="⬇️ Download JSON Recipe", 
                data=recipe_json, 
                file_name="pipeline_recipe.json", 
                mime="application/json"
            )
        else:
            st.info("Pipeline đang trống, không có gì để export.")
            
    with rcol2:
        st.markdown("**Import Pipeline Recipe**")
        uploaded_recipe = st.file_uploader("Upload JSON Recipe", type=["json"], key="recipe_uploader")
        if uploaded_recipe is not None:
            if st.button("⬆️ Load Recipe"):
                try:
                    recipe = json.load(uploaded_recipe)
                    st.session_state.current_pipeline = recipe
                    st.success("Nạp Recipe thành công! Quay lại Tab 1 để xem.")
                except Exception as e:
                    st.error(f"Lỗi đọc file JSON: {e}")
