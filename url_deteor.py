from transformers import pipeline


print("Loading the model (Takes a moment for the first time)...")

classifier = pipeline("text-classification",
                      model="Eason918/malicious-url-detector-v2",
                      truncation=True)


#2 Define urls you want to check

test_urls = [
    "https://www.google.com",  # Safe URL
    "http://malicious-example.com",  # Malicious URL
    "https://www.example.com",  # Safe URL
    "http://phishing-site.net",  # Malicious URL
    "https://www.github.com",  # Safe URL
    "http://fake-login-page.org",  # Malicious URL
]

#Scan results
print("Scanning URLs for malicious content...")

for url in test_urls:
    result = classifier(url)[0]
    print(f"URL: {url}")
    print(f"Verdict: {result['label']} (Confidence: {result['score']:.2%})\n")

