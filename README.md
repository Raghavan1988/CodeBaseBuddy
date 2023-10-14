# CodeBaseBuddy


Presentation : https://docs.google.com/presentation/d/1snzmLBxwP_0WuHEyJUdqvj_8iVpj_isCiKgMzaR7EWE/edit?usp=sharing


# Instructions
1. pip install -r requirements.txt
2. Clone the github repository of your choice (eg. https://github.com/PromtEngineer/localGPT)
3. git clone https://github.com/PromtEngineer/localGPT
4.  export OPENAI_API_KEY= <your key on the terminal> and also in the code.
5. Build the index:**python build_embeddings.py <name> <path to local github branch>** eg. python build_embeddings.py localGPT /home/raghavan/localGPT/
6. ![Screenshot from 2023-10-14 00-19-13](https://github.com/Raghavan1988/CodeBaseBuddy/assets/493090/5bd63e8b-52f6-483d-8a06-492d50cd2fff)
7. Query for your task
8. (base) raghavan@raghavan-Blade-18-RZ09-0484:~/CodeBaseBuddy$ **python search.py "which files should i change and how should i add support to new LLM Falcon 80b" 5 localGPT**
9. ![Screenshot from 2023-10-14 00-21-33](https://github.com/Raghavan1988/CodeBaseBuddy/assets/493090/03a1f3b5-f939-43bb-b1c9-efcec6548c22)
10. Remember LLMs hallucinate :-)

Loom Video: https://www.loom.com/share/348d46575c2f42a68b8fa82d879f40aa?sid=45fc13ea-55a1-40b0-915b-dc8c0adbf4b9
