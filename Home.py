import streamlit as st
from PIL import Image
import base64
    
# Injecting CSS for custom fonts, background, and layout
st.markdown(
    """

   <style>
    .main {
        background-color: #D2E0FB
    }
    </style>

    <style>
        @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap');
    
    /* Button styles */

    .stButton > button {
        background-color: #7695FF;
        border-radius: 12px;
        padding: 10px 24px;
        color: white;
        font-size: 16px;
        font-family: 'Quicksand', sans-serif;
        font-weight: bold;
    }

    /* Hover effect for buttons */
    .stButton > button:hover {
        background-color: #125B9A;
        color: white;
    }
    /* Header style */
    header {
        background-color: #227B94;
        padding: 15px;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    header img {
        max-height: 60px;
        margin-right: 20px;
    }

    header h1 {
        font-family: 'Libre Baskerville', sans-serif;
        font-size: 2.3em;
        color: white;
        text-shadow: 1px 1px 2px #000000;
        margin: 0;
    }
    
    h3 {
        font-family: 'Libre Baskerville', sans-serif;
        font-size: 2em;
        color: #227B94;
        justify-content: center;
    }

    p {
        font-family: "Quicksand", system-ui;
        font-size: 1em;
        line-height: 1.5em;
        color: #333333;
    }

    /* Expander styling */
        .expanderHeader {
            font-family: "Quicksand", sans-serif;
            font-size: 1em;
            color: #333333;
        }
     .expanderContent {
            font-family: "Quicksand", sans-serif;
            font-size: 1em;
            color: #333333;
            margin: 10px 0;
        }
    /* List styles */
        ul {
            list-style-type: disc;
            margin-left: 20px;
        }

        .heat-index {
            font-style: normal;
            color: #333333;
        }

        .water-pollution {
            font-style: normal;
            color: #333333;
        }

        .population-growth {
            font-style: normal;
            color: #333333;
        }

        .agricultural-activities {
            font-style: normal;
            color: #333333;
        }

        .climate-change {
            font-style: normal;
            color: #333333;
        }


    /* Footer styles */
    footer {
        font-family: "Quicksand", system-ui;
        text-align: center;
        padding: 15px;
        background-color: #16325B;
        color: white;
        margin-top: 20px;
        font-size: 1em;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Function to convert the image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Convert your logo image to base64
logo_base64 = image_to_base64("logo.png")

st.markdown(f"""
<header style="display: flex; align-items: center;">
    <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="max-height: 60px; margin-right: 20px;">
    <h1>Water Scarcity Forecast</h1>
</header>
""", unsafe_allow_html=True)

# Tabs for navigation
tab1, tab2 = st.tabs(["Home", "Factors"])

with tab1:
    # Introduction and images
    with st.container():
     col1, col2 = st.columns([1, 1.3])

     with col1:
            st.image("water_scarcity.jpg", caption="© World Wildlife", width=300)

    with col2:
            st.markdown("""
            ### What is Water Scarcity?
            Water scarcity is defined as the state of imbalance between water demand and supply, where water availability is compromised. This has been a pressing issue as faced by many countries due to several factors but water scarcity factors remain under explored. Additionally, water scarcity is the lack of water resources to meet the standard water demand. There are two types of water scarcity: physical, and economic water scarcity. Physical water scarcity is where there is not enough water to meet all demands.
            """)

    # Global and local water scarcity details
    with st.container():
        col1, col2 = st.columns([1.3, 1])

        with col1:
                st.markdown("""
            ### Water Scarcity in the World
            Water scarcity is due to the lack of sufficient water and lack of access to safe water supplies. The phenomena has been affecting the lives of many population across the globe. Moreover, according to Kuzma (2023), this problem results in water stress, a condition that many countries experience where there is an imbalance between the supply and demand of water."
                            """)
            
        with col2:
            st.image("world.jpg", caption="© Unicef", width=300)
        with st.expander("Learn More"):
                            st.markdown("""
            Furthermore, access to water remains a development issue across the continent, specifically in Africa, with one in three Africans facing water scarcity. However, despite the action against the water scarcity, the phenomenon remains a massive issue across the world. One of the factors of water scarcity is the shortage in the supply of consumable water brought by several factors. Yet, the water scarcity still prevails due to the climate change, the environmentally harmful and water-intensive activities of humans (Ingrao et al., 2023).
            """)
            
    # Philippine water scarcity details
    with st.container():
        col1, col2 = st.columns([1, 1.3])

        with col1:
            st.image("philippine.jpg", caption="© Cebu City Councilor Joel Garganera", width=300)

        with col2:
            st.markdown("""
            ### Water Scarcity in the Philippines
            The Philippines is no excuse from the pervasiveness of water scarcity as statistics recorded in year 2020 implied that 57 million out of 109 million people (52%) lack access to consumable and safely managed water and 39% of those suffers from the absence of proper water sanitation systems according to Water.org in 2020. 
            """)
       
        with st.expander("Learn More"):
            st.markdown("""
        Metro Manila is one specific region where water scarcity is felt with 13 million of its population experiencing water shortages stated by the National Public Radio in 2023. In more recent events, according to the report of the Philippine Atmospheric, Geophysical and Astronomical Services Administration (PAGASA), CALABARZON region has been facing droughts since 2024 started. Cavite was hit with drought, Rizal with dry spell, and Laguna and Batangas with dry conditions base on the study of Odong in 2024. It was also reported by the GMA news that Barangay Bucandala 3, an area in Imus Cavite, experienced an unsteady water supply for 6 months in 2022. This indicates the urgency of recognizing such nationwide phenomenon.
        """)

    st.markdown("""
    ### Factors affecting water scarcity
    The notable factors which contribute to water scarcity are heat index, water pollution, population growth, agricultural activities, and climate change. 
    <ul>
    <li class="heat-index">
        <strong>Heat index</strong> - as it rises, the probability of drought occuring also increases.  The demands also increases especially in El Niño season. Thus lessened the available water needed.
    </li>
    <li class="water-pollution">
        <strong>Water pollution</strong> -  the main cause of it was human activities such as throwing trashes everywhere. This contaminates the bodies of water or available freshwater. As a result, the water cannot be safely consumed or used, descreasing the available water for us to consume.
    </li>
    <li class="population-growth">
        <strong>Population growth</strong> - the increase of population also increases the water demand needed to supply each sectors. This contributes to water stress.
    </li>
    <li class="agricultural-activities">
        <strong>Agricultural activities</strong> - contributes to the prevalence of water scarcity. In was recorded that 60% of the used fresh water by agricultural activities is wasted due to leaky irrigation systems, inefficient application methods, and the cultivation of crops thirsty for the environment they were grown in. According to the data of World Wide Fund (WWF), year 2024.
    </li>
    <li class="climate-change">
        <strong>Climate change</strong>  that which exacerbates this phenomena. Human activities that produces more carbon dioxide  continuously affect weather patterns as these compounds intoxicates the atmosphere, as result, drought seasons will be more occurring and glaciers will vanish, limiting the supply of water, according to World Wide Fund (WWF) also as of 2024.
    </li>
    </ul>
    """, unsafe_allow_html=True)

with tab2:

# Information about the factors
    st.markdown("""        
    ### Heat Index

    Also known as the apparent temperature, it indicates the level of discomfort that the person is expected to experience due to the combined effects of air temperature and humidity. The heat index is directly proportional to the humidity. Therefore, as humidity increases, the heat index also increases, and when the humidity decreases, the heat index also decreases. 
    """)

    st.markdown("""    
    ### Rainfall

    The total amount of rain falling in a specific area within a particular time frame.
    """)

    st.markdown("""    
    ### Total Water Withdrawal

    Total water withdrawal is the yearly freshwater withdrawal in which the freshwater is taken from ground or surface water sources and can be permanent or temporary. Then, transferred to a place of use.
    """)

    st.markdown("""    
    ### Regression Analysis

    # Create the button for Regression Analysis

if st.button("Regression Analysis"):
    st.experimental_set_query_params(page="Regression Analysis")
    st.write("Navigating to Regression Analysis page...")
    st.experimental_rerun()
    
    """)

# Custom footer
st.markdown("""
<footer>
   Gen. Juan Castaneda Senior High School | © 2025
</footer>
""", unsafe_allow_html=True)
