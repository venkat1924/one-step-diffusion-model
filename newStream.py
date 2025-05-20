import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import math


try:
    from teacherRectifiedFlow import UNet as TeacherUNet, RectifiedFlow, SinusoidalPositionEmbeddings as TeacherSinusoidalEmbeddings, AdaGN, ResidualBlock as TeacherResidualBlock, AttentionBlock as TeacherAttentionBlock
    from fourStepStudent import StudentModel, SinusoidalPositionEmbeddings as StudentSinusoidalEmbeddings, TimeAwareGroupNorm, StudentBlock
except ImportError as e:
    st.error(f"Failed to import model definitions. Ensure 'teacherRectifiedFlow.py' and 'fourStepStudent.py' are in the same directory as the Streamlit app. Error: {e}")
    st.stop()


def exists(x):
    return x is not None

def default(val, d):
    return val if exists(val) else d() if callable(d) else d

@st.cache_resource
def load_teacher_model(checkpoint_path, device):
    teacher_unet = TeacherUNet(
        in_channels=3,
        out_channels=3,
        init_channels=32,
        time_emb_dim=256,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        num_heads=4,
        attn_resolutions=(16,)
    ).to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            teacher_unet.load_state_dict(checkpoint['model_state_dict'])
        else:
            teacher_unet.load_state_dict(checkpoint)
    except Exception as e:
        st.error(f"Error loading teacher UNet weights from {checkpoint_path}: {e}")
        return None

    rectified_flow_model = RectifiedFlow(teacher_unet, timesteps=1000).to(device)
    rectified_flow_model.eval()
    return rectified_flow_model

@st.cache_resource
def load_student_model(checkpoint_path, device):
    student_model_instance = StudentModel(
        in_ch=3,
        out_ch=3,
        init_ch=24,
        time_dim=256
    ).to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state' in checkpoint:
            student_model_instance.load_state_dict(checkpoint['model_state'])
        else:
            student_model_instance.load_state_dict(checkpoint)
    except Exception as e:
        st.error(f"Error loading student model weights from {checkpoint_path}: {e}")
        return None

    student_model_instance.eval()
    return student_model_instance

st.set_page_config(layout="wide", page_title="Rectified Flow Image Generation")
st.title("🖼️ Rectified Flow & Student Model Image Generation")

st.markdown("""
Welcome to the Image Generation interface using Rectified Flow models!
You can choose between the original **Teacher (U-Net based Rectified Flow)** model or
the distilled **Student (4-step)** model to generate images.
The required checkpoint files (`rectified_model_final.pth` for Teacher, `latest.pth` for Student)
must be in the same directory as this script.
""")

st.sidebar.header("⚙️ Controls")
model_choice = st.sidebar.selectbox("Choose Model", ["Teacher (Rectified Flow)", "Student (4-step)"])

if torch.cuda.is_available():
    device = torch.device("cuda")
    st.sidebar.info("✅ CUDA is available. Using CUDA.")
else:
    device = torch.device("cpu")
    st.sidebar.info("ℹ️ CUDA not available. Using CPU.")

TEACHER_CHECKPOINT_PATH = "rectified_flow_model_final.pth"
STUDENT_CHECKPOINT_PATH = "latest.pth"

if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_type_loaded' not in st.session_state:
    st.session_state.model_type_loaded = None
if 'current_checkpoint_attempted' not in st.session_state:
    st.session_state.current_checkpoint_attempted = ""


requested_checkpoint_path = ""
if model_choice == "Teacher (Rectified Flow)":
    requested_checkpoint_path = TEACHER_CHECKPOINT_PATH
else:
    requested_checkpoint_path = STUDENT_CHECKPOINT_PATH


if st.session_state.model_type_loaded != model_choice or st.session_state.current_checkpoint_attempted != requested_checkpoint_path:
    st.session_state.model = None
    st.session_state.model_type_loaded = None
    st.session_state.current_checkpoint_attempted = requested_checkpoint_path

    if not os.path.exists(requested_checkpoint_path):
        st.sidebar.error(f"Checkpoint file not found: {requested_checkpoint_path}")
    else:
        with st.spinner(f"Loading {model_choice} from {requested_checkpoint_path}..."):
            loaded_model = None
            if model_choice == "Teacher (Rectified Flow)":
                loaded_model = load_teacher_model(requested_checkpoint_path, device)
            else:
                loaded_model = load_student_model(requested_checkpoint_path, device)

            if loaded_model is not None:
                st.session_state.model = loaded_model
                st.session_state.model_type_loaded = model_choice
                st.sidebar.success(f"{model_choice} loaded!")
            else:
                st.sidebar.error(f"Failed to load {model_choice}.")

model = st.session_state.model


if model is not None and st.session_state.model_type_loaded == model_choice:

    st.sidebar.subheader("Generation Parameters")
    num_images = st.sidebar.slider("Number of Images to Generate", 1, 64, 16, key=f"num_images_{model_choice}")
    batch_size_gen = st.sidebar.number_input("Batch Size for Generation", 1, num_images, min(8, num_images), key=f"batch_size_{model_choice}")
    seed = st.sidebar.number_input("Seed for Noise", value=44, min_value=0, step=1, key=f"seed_{model_choice}")

    num_steps_teacher = 100
    if st.session_state.model_type_loaded == "Teacher (Rectified Flow)":
        num_steps_teacher = st.sidebar.slider("Sampling Steps (Teacher)", min_value=4, max_value=1000, value=100, step=4, key=f"steps_teacher_{model_choice}")
        st.sidebar.markdown("<small>Teacher model was trained evaluating with 100 and 4 steps.</small>", unsafe_allow_html=True)

    if st.sidebar.button("🚀 Generate Images", use_container_width=True, key=f"generate_button_{model_choice}"):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

        all_images_list = []
        generation_model_name = st.session_state.model_type_loaded
        with st.spinner(f"Generating {num_images} image(s) with {generation_model_name}... Please wait."):
            for i in range(0, num_images, batch_size_gen):
                current_batch_size = min(batch_size_gen, num_images - i)
                noise = torch.randn((current_batch_size, 3, 32, 32), device=device)

                if st.session_state.model_type_loaded == "Teacher (Rectified Flow)":
                    with torch.no_grad():
                        generated_samples = model.sample(noise=noise,
                                                         batch_size=current_batch_size,
                                                         num_steps=num_steps_teacher)
                else:
                    x_student = noise.clone()
                    with torch.no_grad():
                        for step_idx in range(4):
                            current_t_values = torch.full((current_batch_size,), step_idx / 4.0, device=device)
                            pred_target_image = model(x_student, current_t_values)
                            alpha = (step_idx + 1.0) / 4.0
                            x_student = (1.0 - alpha) * x_student + alpha * pred_target_image
                    generated_samples = x_student

                generated_samples = generated_samples.clamp(-1, 1)
                all_images_list.append(generated_samples.cpu())

        if all_images_list:
            final_images_tensor = torch.cat(all_images_list, dim=0)
            grid_nrow = int(math.ceil(np.sqrt(num_images))) if num_images > 0 else 1
            grid = torchvision.utils.make_grid(final_images_tensor, nrow=grid_nrow, normalize=True, scale_each=True)

            st.subheader("🖼️ Generated Images")
            st.image(grid.permute(1, 2, 0).numpy(), width=400)

            grid_pil = transforms.ToPILImage()(grid)
            from io import BytesIO
            buf = BytesIO()
            grid_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            filename_model_type_cleaned = st.session_state.model_type_loaded.replace(' ','_').replace('(', '').replace(')', '')
            num_steps_for_filename = num_steps_teacher if st.session_state.model_type_loaded == "Teacher (Rectified Flow)" else 4

            st.download_button(
                label="Download Image Grid",
                data=byte_im,
                file_name=f"{filename_model_type_cleaned}_seed{seed}_steps{num_steps_for_filename}.png",
                mime="image/png",
                use_container_width=True
            )
else:
    st.markdown("---")
    st.warning(f"**Model ('{model_choice}') is not loaded.**")
    st.info(
        f"The application is trying to load `{requested_checkpoint_path}`.  \n"
        f"Please ensure the required checkpoint files are present in the script's directory:  \n"
        f"- For Teacher model: `{TEACHER_CHECKPOINT_PATH}`  \n"
        f"- For Student model: `{STUDENT_CHECKPOINT_PATH}`  \n"
        f"If the file exists but loading fails, an error message should appear in the sidebar."
    )


st.markdown("---")
st.header("ℹ️ Model & Generation Details")

with st.expander("What is Rectified Flow?"):
    st.markdown("""
    Rectified Flow is a type of generative model that learns to transform a simple noise distribution (e.g., Gaussian noise)
    into a complex data distribution (e.g., images) by learning an Ordinary Differential Equation (ODE).
    The "rectification" process aims to create straighter, more stable paths for this transformation,
    allowing for efficient and high-quality generation, often in fewer steps than traditional diffusion models.

    The core idea is to model the velocity of a particle moving from a noise sample $z_0$ to a data sample $z_1$ along a straight line $z_t = (1-t)z_0 + t z_1$. The model learns the velocity field $v(z_t, t) \simeq z_1 - z_0$.
    During sampling, we start with $z_1 \simeq \mathcal{N}(0,I)$ (noise at $t=1$) and solve the ODE backwards from $t=1$ to $t=0$ using an ODE solver (like Euler method): $z_{t - \Delta t} = z_t - v(z_t, t)\Delta t$.
    """)

with st.expander("Teacher Model (U-Net based Rectified Flow)"):
    st.markdown(f"""
    The **Teacher Model** directly implements the Rectified Flow concept using a U-Net architecture to predict the velocity required to transform noise into an image.
    - **Architecture**: U-Net with Sinusoidal Position Embeddings for time, Residual Blocks, and Attention.
    - **Training**: Typically trained on datasets like CIFAR-10 to predict the velocity vector.
    - **Sampling**: Generates images by starting with random noise and iteratively applying the learned velocity field to denoise it over a specified number of steps. More steps usually yield better quality but take longer.
    - **Required Checkpoint**: `{TEACHER_CHECKPOINT_PATH}` (must be in the script's directory).
    """)

with st.expander("Student Model (4-step)"):
    st.markdown(f"""
    The **Student Model** is a smaller, faster model designed for very few-step generation (specifically 4 steps). It's often trained via distillation from the Teacher model.
    - **Architecture**: A custom CNN architecture, also time-aware, designed for efficiency.
    - **Training**: Learns to predict images or refined versions from an input $x_t$ at time $t$, often using a combination of losses like MSE and Perceptual Loss.
    - **Sampling**: This interface replicates a 4-step iterative refinement process. The `t` values passed to the student model during these 4 steps are typically $0, 0.25, 0.5, 0.75$.
    - **Required Checkpoint**: `{STUDENT_CHECKPOINT_PATH}` (must be in the script's directory).
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Project Title:**  
*RVCE_24VI48RV - GenAI: One-step Diffusion Model Using Distillation Techniques for Text-to-Image*

**Under the PRISM Program by:**  
*Samsung Research Institute, Bangalore*

**Created by:**  
- Anumaneni Venkat Balachandra  
- Sravya D  
- Shreyashwini R 
- Sai Varun Konda  

**Under the guidance of:**  
- Dr. Dendi Sathya Veera Reddy  
- Prof. Ganashree M  
- Prof. Rajani Katiyar
""")
