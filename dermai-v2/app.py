import streamlit as st
import torch
from PIL import Image
import numpy as np
import time
import streamlit.components.v1 as components

# Local imports
from engine import load_engine, predict
from rag import get_rag_concierge

st.set_page_config(page_title="DermAI V2 - Multi-Modal Virtual Biopsy", layout="wide")

# --- Initialize Session State ---
if 'engine' not in st.session_state:
    with st.spinner("Loading Vision Transformer Engine..."):
        st.session_state.engine, st.session_state.preprocess, st.session_state.device = load_engine()
        st.session_state.rag = get_rag_concierge()

# --- Sidebar: Multi-Modal Ingestion ---
st.sidebar.header("Stage 1: Multi-Modal Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Skin Lesion Image", type=["jpg", "png", "jpeg"])

age = st.sidebar.slider("Patient Age", 0, 100, 45)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
anatomy = st.sidebar.selectbox("Anatomical Location", ["Head/Neck", "Trunk", "Upper Extremity", "Lower Extremity", "Acral"])
family_history = st.sidebar.checkbox("Family History of Melanoma")

# --- Main Dashboard ---
st.title("DermAI: AI-Powered 'Virtual Biopsy' & 3D Digital Twin")
st.markdown("---")

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="Uploaded Lesion", use_container_width=True)
        
    with col2:
        st.subheader("Stage 2: The Virtual Biopsy Engine")
        if st.button("Run Multi-Modal Analysis"):
            # Normalize metadata for the engine
            metadata_list = [
                age / 100.0,
                1.0 if gender == "Male" else 0.0,
                ["Head/Neck", "Trunk", "Upper Extremity", "Lower Extremity", "Acral"].index(anatomy) / 4.0,
                1.0 if family_history else 0.0
            ]
            
            with st.spinner("Analyzing cross-attention features..."):
                risk_score, mutation_prob = predict(
                    st.session_state.engine, 
                    st.session_state.preprocess, 
                    st.session_state.device, 
                    image, 
                    metadata_list
                )
                
            st.write(f"**Metastatic Risk Score:** {risk_score*100:.2f}%")
            st.write(f"**BRAF V600E Mutation Probability:** {mutation_prob*100:.2f}%")
            
            if mutation_prob > 0.5:
                st.success("Genomic Profile: BRAF V600E Positive Detected (In-Silico)")
                st.session_state.mutation_status = "BRAF_V600E"
            else:
                st.info("Genomic Profile: Wild-Type / Low Mutation Probability")
                st.session_state.mutation_status = "Wild_Type"
            
            st.session_state.risk_score = risk_score
            st.session_state.analysis_done = True

    if st.session_state.get('analysis_done'):
        st.markdown("---")
        st.subheader("Stage 3: Data-Driven Treatment Recommender")
        
        if st.session_state.mutation_status == "BRAF_V600E":
            st.info("Statistical Analysis based on COMBI-d/v Trials (Melanoma)")
            st.write("""
            **Recommended Protocol:** Targeted Therapy (Dabrafenib + Trametinib)
            - **Objective Response Rate (ORR):** 68% volume reduction expected.
            - **Complete Response (CR):** 19% probability.
            - **5-Year Overall Survival:** 34% (General population).
            """)
            
            st.markdown("---")
            st.subheader("Stage 4: 3D Digital Twin Simulation")
            
            day_slider = st.select_slider("Treatment Timeline (Days)", options=list(range(0, 91, 10)), value=0)
            
            # --- Advanced Three.js Digital Twin Overhaul ---
            shrink_factor = 1.0 - (0.68 * (day_slider / 90.0))
            smoothing_factor = (day_slider / 90.0) # 0 to 1
            
            three_js_code = f"""
            <div id="container" style="width: 100%; height: 500px; background-color: #050505; border-radius: 10px; overflow: hidden;"></div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
            <script>
                const container = document.getElementById('container');
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(45, container.clientWidth / 500, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
                renderer.setSize(container.clientWidth, 500);
                renderer.setPixelRatio(window.devicePixelRatio);
                container.appendChild(renderer.domElement);

                // --- Lighting (Clinical/Studio Setup) ---
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
                scene.add(ambientLight);
                
                const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
                mainLight.position.set(5, 10, 7.5);
                scene.add(mainLight);

                const rimLight = new THREE.PointLight(0xff0000, 0.5); // Simulated Subsurface Scattering glow
                rimLight.position.set(-5, -5, -5);
                scene.add(rimLight);

                // --- Reference Grid ---
                const grid = new THREE.GridHelper(10, 20, 0x444444, 0x222222);
                grid.position.y = -2;
                scene.add(grid);

                // --- Procedural Irregular Tumor ---
                // We use displacement to make it irregular
                const geometry = new THREE.IcosahedronGeometry(1.5, 32); 
                const position = geometry.attributes.position;
                const vector = new THREE.Vector3();
                
                // Save original positions for shrinking/smoothing
                const originalPositions = position.array.slice();

                function updateTumorShape(shrink, smooth) {{
                    for (let i = 0; i < position.count; i++) {{
                        vector.fromArray(originalPositions, i * 3);
                        
                        // Add procedural noise (deterministic for this sample)
                        const noise = (Math.sin(vector.x * 5) * Math.cos(vector.y * 5) * Math.sin(vector.z * 5)) * 0.3;
                        
                        // As treatment progresses (smooth increases), the noise dampens
                        const currentNoise = noise * (1.0 - smooth);
                        
                        const factor = shrink + currentNoise;
                        vector.multiplyScalar(factor);
                        position.setXYZ(i, vector.x, vector.y, vector.z);
                    }}
                    position.needsUpdate = true;
                    geometry.computeVertexNormals();
                }}

                const material = new THREE.MeshStandardMaterial({{ 
                    color: 0x6e1a1a, 
                    roughness: 0.3, 
                    metalness: 0.1,
                    flatShading: false,
                    emissive: 0x220000,
                    emissiveIntensity: 0.2
                }});

                const tumor = new THREE.Mesh(geometry, material);
                scene.add(tumor);

                updateTumorShape({shrink_factor}, {smoothing_factor});

                camera.position.set(4, 2, 6);
                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;

                function animate() {{
                    requestAnimationFrame(animate);
                    tumor.rotation.y += 0.005;
                    controls.update();
                    renderer.render(scene, camera);
                }}
                animate();

                window.addEventListener('resize', () => {{
                    camera.aspect = container.clientWidth / 500;
                    camera.updateProjectionMatrix();
                    renderer.setSize(container.clientWidth, 500);
                }});
            </script>
            """
            components.html(three_js_code, height=520)
            
            st.caption(f"3D Morphological Model: Tumor Volume at Day {day_slider} ({int(shrink_factor*100)}% of initial size)")
            
            # --- RAG Concierge ---
            st.markdown("---")
            st.subheader("RAG Concierge: Clinical Guideline Citations")
            query = f"What is the standard of care for BRAF V600E melanoma and what are the expected survival rates for targeted therapy?"
            if st.button("Query NCCN Guidelines"):
                results = st.session_state.rag.search(query)
                st.markdown(results)
        else:
            st.warning("Genetic profile does not match targeted therapy criteria. Consult NCCN guidelines for Immunotherapy options.")
            if st.button("Query Guidelines for Wild-Type"):
                results = st.session_state.rag.search("First-line therapy for BRAF wild-type metastatic melanoma")
                st.markdown(results)

else:
    st.info("Please upload a lesion image in the sidebar to begin the Virtual Biopsy process.")
    st.image("https://images.unsplash.com/photo-1584634731339-252c581abfc5?q=80&w=2000&auto=format&fit=crop", caption="DermAI Digital Twin Framework", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("DermAI V2.0 - Developed for Precision Oncology")
