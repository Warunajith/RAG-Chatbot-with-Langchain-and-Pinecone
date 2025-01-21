import requests
import streamlit as st
import base64

st.title("Conversational RAG ChatBot with Langchain By Warunajith")

# Set your FastAPI endpoint URL here
FASTAPI_ENDPOINT = "http://localhost:5000/api/"


# Make a request to FastAPI to get a session token
def get_session_token():
    response = requests.post(FASTAPI_ENDPOINT + "generate-session")
    if response.status_code == 200:
        print(f"Session Token Generated: {response.json()['session_token']}")
        return response.json()['session_token']
    else:
        st.error("Failed to generate session token.")
        return None


# Store the session token in session state
if 'session_token' not in st.session_state:
    st.session_state.session_token = get_session_token()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Allow the user to upload a PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Check if a file has been uploaded
if uploaded_file is not None:
    # Convert PDF file to base64 for sending in the API call
    pdf_data = uploaded_file.read()  # Read the file in binary
    pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')  # Convert to base64

    st.success("PDF uploaded successfully.")

# Get user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    headers = {'Authorization': f"Bearer {st.session_state.session_token}"}

    if uploaded_file:
        files = {
            "file": uploaded_file.getvalue()
        }

        # Prepare the data to send in the API request
        data = {
            "question": st.session_state.messages[-1]["content"],
            
        }

        # Send request to FastAPI endpoint
        response = requests.post(
            FASTAPI_ENDPOINT + "rag-chatbot",
            files=files,
            data=data,
            headers=headers
        )

        # Get response from FastAPI
        if response.status_code == 200:
            assistant_reply = response.json().get("response", "")

            print(f"Assistant Reply: {assistant_reply}")

            with st.chat_message("assistant"):
                st.markdown(assistant_reply)

            # Save assistant's response
            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        else:
            st.error("Error connecting to the API.")
    else:
        st.error("Please upload a PDF file.")
