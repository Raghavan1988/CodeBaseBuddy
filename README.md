# CodeBaseBuddy

Welcome to CodeBaseBuddy! This tool helps you navigate and understand your codebase better. For a visual walkthrough, check out our [Presentation](https://lablab.ai/event/open-interpreter-hackathon/githubbuddy/codebasebuddy)
 and [Loom](https://www.loom.com/share/348d46575c2f42a68b8fa82d879f40aa?sid=45fc13ea-55a1-40b0-915b-dc8c0adbf4b9) video

 DZONE article: [link](https://dzone.com/articles/code-search-using-retrieval-augmented-generation)

## Getting Started

Follow these steps to get CodeBaseBuddy up and running:

1. Install the required packages: `pip install -r requirements.txt`
2. Clone the GitHub repository of your choice. For example: `git clone https://github.com/PromtEngineer/localGPT`
3. Set your OpenAI API key in your terminal and in the code: `export OPENAI_API_KEY=<your key>`
4. Build the index with the following command: `python build_embeddings.py <name> <path to local github branch>`. For example: `python build_embeddings.py localGPT /home/raghavan/localGPT/`

## Usage

Once you've set up CodeBaseBuddy, you can start querying for your tasks. Here's an example:

```bash
(base) raghavan@raghavan-Blade-18-RZ09-0484:~/CodeBaseBuddy$ python search.py "which files should i change and how should i add support to new LLM Falcon 80b" 5 localGPT
```

## Sample screenshots
[Screenshot from 2023-10-14 00-19-13](https://github.com/Raghavan1988/CodeBaseBuddy/assets/493090/5bd63e8b-52f6-483d-8a06-492d50cd2fff)
[Screenshot from 2023-10-14 00-21-33](https://github.com/Raghavan1988/CodeBaseBuddy/assets/493090/03a1f3b5-f939-43bb-b1c9-efcec6548c22)
Remember LLMs hallucinate :-)
