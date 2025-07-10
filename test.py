import requests
import time


def test_rag_api():
    base_url = "http://192.168.1.32:8888"

    try:
        health_response = requests.get(f"{base_url}/")
        print("Health Check:", health_response.json())
    except Exception as e:
        print("Health check failed:", str(e))
        return


    test_questions = [f"""i need you to help me get the recommendation for this case findings the petrous , cavernous portions of right ICA.
•	Atherosclerotic changes of both common , external and internal carotid arteries are seen  manifested as intimal thickening with non-calcified atheromatous plaque 
"""

    ]

    for question in test_questions:
        try:
            print("\nSending question:", question)
            response = requests.post(
                f"{base_url}/ask",
                json={"question": question},
                timeout=30  # 30 seconds timeout
            )

            if response.status_code == 200:
                print("Answer:", response.json()['answer'])
                print("source:", response.json()['source'])
            else:
                print(f"Error {response.status_code}:", response.json())

        except requests.exceptions.Timeout:
            print("Request timed out")
        except Exception as e:
            print("Error:", str(e))

        time.sleep(2)  # Wait 2 seconds between questions


if __name__ == "__main__":
    test_rag_api()
