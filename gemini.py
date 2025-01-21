import google.generativeai as genai

genai.configure(api_key="")

model = genai.GenerativeModel('gemini-1.0-pro-latest')
response = model.generate_content("siapa presiden indonesia saat ini")
print(response.text)