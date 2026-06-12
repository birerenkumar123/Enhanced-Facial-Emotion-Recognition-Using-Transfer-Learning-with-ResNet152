import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf

# Set page config
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

# Enhanced custom CSS for styling
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-family: 'Segoe UI', sans-serif; }
    .main-header { font-size: 3.5rem; color: white; text-align: center; margin: 2rem 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    .sub-header { font-size: 1.8rem; color: #f8f9fa; text-align: center; margin-bottom: 2.5rem; }
    .card { background: rgba(255, 255, 255, 0.95); border-radius: 15px; padding: 2rem; margin: 1rem 0; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2); }
    .upload-section { border: 3px dashed #667eea; border-radius: 15px; padding: 3rem; text-align: center; background: rgba(255, 255, 255, 0.1); }
    .stButton>button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 50px; padding: 1rem 2rem; font-weight: bold; border: none; }
    .result-container { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 2rem; color: white; text-align: center; }
    .emotion-result { font-size: 2.5rem; font-weight: bold; }
    .footer { text-align: center; margin-top: 3rem; color: rgba(255, 255, 255, 0.8); }
    .stProgress > div > div { background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%) !important; }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">😊 Facial Emotion Recognition</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">Detect emotions using cutting-edge AI and computer vision</h2>', unsafe_allow_html=True)
st.markdown('<div class="card"><p style="font-size: 1.2rem; text-align: center;">Upload a clear photo of a face and our advanced AI model will analyze the facial expression to detect the emotion.</p></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# AGGRESSIVE COMPATIBILITY PATCH
# ----------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    # More robust model path detection with case-insensitive search
    model_filename = 'emotion_model.keras'
    expected_dir = 'Enhanced-Facial-Emotion-Recognition-Using-Transfer-Learning-with-ResNet152'
    
    # First try the exact path
    model_path = os.path.join(expected_dir, model_filename)
    
    # Check alternative paths if the first one doesn't exist
    if not os.path.exists(model_path):
        # Try with different path separators or naming conventions
        alternative_paths = [
            'Enhanced-Facial-Emotion-Recognition-Using-Transfer-Learning-with-ResNet152\\emotion_model.keras',
            os.path.join('Enhanced-Facial-Emotion-Recognition-Using-Transfer-Learning-with-ResNet152', 'emotion_model.keras')
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                break
        
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file not found at: {model_path}")
        # List contents of the directory to help with debugging
        dir_path = 'Enhanced-Facial-Emotion-Recognition-Using-Transfer-Learning-with-ResNet152'
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            st.error(f"Files in directory: {files}")
        else:
            st.error(f"Directory does not exist: {dir_path}")
        return None

    # 1. Backup the original from_config method
    original_from_config = tf.keras.layers.Layer.from_config

    # 2. Define the patch that aggressively cleans the config dict
    @classmethod
    def patched_from_config(cls, config):
        # Make a copy to avoid modifying the original
        config = config.copy()
        
        # --- FIX 1: REMOVE KERAS 3 DEFAULTS (sparse, ragged) ---
        # These arguments cause the "Keyword argument not understood" error in Keras 2
        if 'sparse' in config:
            del config['sparse']
        if 'ragged' in config:
            del config['ragged']
        
        # Also remove other problematic keys
        if 'autocast' in config:
            del config['autocast']
        if 'trainable' in config:
            del config['trainable']

        # --- FIX 2: DTYPE (Dict -> String) ---
        # Keras 3 uses a dict for dtype, Keras 2 wants a string
        if 'dtype' in config and isinstance(config['dtype'], dict):
            try:
                config['dtype'] = config['dtype']['config']['name']
            except (KeyError, TypeError):
                config['dtype'] = 'float32'
        
        # --- FIX 3: BATCH_SHAPE (Convert to input_shape) ---
        if 'batch_shape' in config:
            batch_shape = config.pop('batch_shape')
            # If this is an InputLayer, we must manually set input_shape
            if cls.__name__ == 'InputLayer' and 'input_shape' not in config and batch_shape:
                if isinstance(batch_shape, (list, tuple)) and len(batch_shape) > 0:
                    config['input_shape'] = batch_shape[1:]
        
        # --- FIX 4: HANDLE OTHER POTENTIALLY PROBLEMATIC KEYS ---
        # Remove any other keys that might cause issues
        unsupported_keys = ['ragged', 'sparse', 'autocast', 'trainable', 'synchronization', 'aggregation']
        for key in unsupported_keys:
            if key in config:
                del config[key]

        # Call the original method with the CLEANED config
        try:
            return original_from_config(config)
        except Exception as e:
            # If we still get an error, try to load with minimal config
            st.warning(f"Warning in patched_from_config for {cls.__name__}: {str(e)}")
            minimal_config = {}
            # Try to preserve essential parameters
            for key in ['name', 'dtype', 'input_shape']:
                if key in config:
                    minimal_config[key] = config[key]
            try:
                return original_from_config(minimal_config)
            except Exception as e2:
                st.error(f"Failed to load layer with minimal config: {str(e2)}")
                raise e

    # 3. Apply the patch globally
    tf.keras.layers.Layer.from_config = patched_from_config

    try:
        # Load model with custom_objects just in case, but the patch handles the heavy lifting
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Critical Error loading model: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        
        # Try alternative loading method
        try:
            st.info("🔄 Trying alternative model loading method...")
            model = tf.keras.models.load_model(model_path, compile=False, 
                                             custom_objects={'from_config': patched_from_config})
            st.success("✅ Model loaded successfully with alternative method!")
        except Exception as e2:
            st.error(f"❌ Alternative loading method also failed: {str(e2)}")
            model = None
    finally:
        # 4. Revert the patch immediately after loading
        tf.keras.layers.Layer.from_config = original_from_config

    # Compile manually if successful
    if model is not None:
        try:
            model.compile(optimizer='Adamax', loss='categorical_crossentropy', metrics=['accuracy'])
        except Exception as e:
            st.warning(f"⚠️ Model compilation warning: {str(e)}")
            # Try alternative compilation approach
            try:
                model.compile(optimizer=tf.keras.optimizers.Adamax(), 
                             loss=tf.keras.losses.CategoricalCrossentropy(), 
                             metrics=['accuracy'])
                st.success("✅ Model compiled successfully with alternative method!")
            except Exception as e2:
                st.error(f"❌ Alternative compilation also failed: {str(e2)}")
                model = None
    
    return model

# ----------------------------------------------------------------------------------
# APP LOGIC
# ----------------------------------------------------------------------------------

emotion_labels = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

st.markdown("""<head><link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet"></head>""", unsafe_allow_html=True)

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((128, 128))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 128, 128, 3)
    return image_array

def predict_emotion(model, image):
    try:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        confidence = np.max(prediction)
        predicted_class = np.argmax(prediction)
        emotion = emotion_labels[predicted_class]
        return emotion, confidence
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        # Return default values in case of error
        return "Unknown", 0.0

col1, col2 = st.columns([2, 3], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #667eea; text-align: center;">📤 Upload Image</h3>', unsafe_allow_html=True)
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a facial image", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        st.markdown('<h4 style="text-align: center; color: #333;">Preview</h4>', unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
        if st.button("🔍 Analyze Emotion", key="predict_btn"):
            with st.spinner("🧠 Analyzing facial expression..."):
                model = load_model()
                
                if model is not None:
                    try:
                        emotion, confidence = predict_emotion(model, image)
                        
                        with col2:
                            st.markdown('<div class="result-container">', unsafe_allow_html=True)
                            st.markdown(f'<h2 class="emotion-result">{emotion}</h2>', unsafe_allow_html=True)
                            st.markdown(f'<p class="confidence">Confidence: {confidence:.2%}</p>', unsafe_allow_html=True)
                            
                            emotion_emojis = {'Ahegao': "😲", 'Angry': "😠", 'Happy': "😄", 'Neutral': "😐", 'Sad': "😢", 'Surprise': "😮"}
                            emotion_messages = {
                                'Ahegao': "You seem to be showing an intense facial expression!",
                                'Angry': "You seem to be feeling angry. Take a deep breath!",
                                'Happy': "You look happy! Keep smiling!",
                                'Neutral': "You have a neutral expression.",
                                'Sad': "You seem sad. Everything will be okay!",
                                'Surprise': "You look surprised! What happened?"
                            }
                            
                            st.markdown(f'<h1 style="font-size: 4rem; text-align: center;">{emotion_emojis.get(emotion, "🤔")}</h1>', unsafe_allow_html=True)
                            st.info(emotion_messages.get(emotion, "Emotion detected."))
                            
                            st.markdown('<p style="text-align: center; font-weight: bold;">Prediction Confidence</p>', unsafe_allow_html=True)
                            st.progress(float(confidence))
                            
                            tips = {
                                'Ahegao': "Express yourself authentically!",
                                'Angry': "Try some relaxation techniques like deep breathing.",
                                'Happy': "Share your joy with others!",
                                'Neutral': "Perfect balance is sometimes the best state.",
                                'Sad': "Remember that difficult times are temporary.",
                                'Surprise': "Embrace life's unexpected moments!"
                            }
                            
                            st.markdown('<div style="margin-top: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.2); border-radius: 10px;">', unsafe_allow_html=True)
                            st.markdown(f'<p style="margin: 0; text-align: center;"><strong>💡 Tip:</strong> {tips.get(emotion, "Stay positive!")}</p>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ Error occurred during prediction: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="info-card"><h3 style="color: #667eea; text-align: center;">ℹ️ How It Works</h3><p>This application uses a state-of-the-art deep learning model based on <strong>ResNet152 architecture</strong>.</p></div>', unsafe_allow_html=True)
st.markdown('<div class="footer"><hr style="border: 1px solid rgba(255,255,255,0.2);"><p>Facial Emotion Recognition App | Powered by TensorFlow & Streamlit</p></div>', unsafe_allow_html=True)