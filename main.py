from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import pickle
import time


# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = tf.keras.models.model_from_json(loaded_model_json)
# loaded_model.load_weights("model.h5")

pickled_model = pickle.load(open('log_reg.pkl', 'rb'))
# pickled_model.predict(X_test)

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2019/04/24/11/27/flowers-4151900_960_720.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 


def classify(out):
    progress_text = "Predicting..."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.0015)
        my_bar.progress(percent_complete + 1, text=progress_text)
    # if(out[0][0]>out[0][1]):
    #     st.subheader('Female')
    # elif(out[0][0]<out[0][1]):
    #     st.subheader('male')
    # else:
    #     st.subheader('Could not define')
    if(out == 0):
        st.subheader('Female')
    elif(out== 1):
        st.subheader('male')
    else:
        st.subheader('Could not define')

def classify_camera(img_file_buffer):
    if (img_file_buffer != None):
        cam = Image.open(img_file_buffer)
        print(type(cam))
        cam = np.resize(cam,(1,2500))   
        out = pickled_model.predict(cam)
        classify(out)


def classify_imported(img):
    img = np.resize(img,(1,2500))
    y = pickled_model.predict(img)
    classify(y)

    


def main():
    st.title('Gender Classification')


    with st.expander("Open Camera"):
        img_file_buffer = st.camera_input("Show your face and Capture")
        if img_file_buffer:
            print(type(img_file_buffer))
            if (st.button("Predict Gender")):
                classify_camera(img_file_buffer)

    with st.expander('Import photo'):
        img = st.file_uploader('import photo')
        if (img):
            st.image(img, caption='imported image', width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
            img = Image.open(img)
            classify_imported(img)

            
    
if __name__ == '__main__':
    main()





