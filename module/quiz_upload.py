def handle_quiz_upload(uploaded_file):
    if uploaded_file:
        filepath = f"./uploads/{uploaded_file.filename}"
        uploaded_file.save(filepath)
        # Do additional processing here if needed
        return uploaded_file.filename
    return "No file received"