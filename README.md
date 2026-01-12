# Github Librarian 📖

Github Librarian is a **RAG-based Semantic Search Engine**, made for educational purposes. Just type in what type of tech you're looking for *(content management systems, web app builders, email sending libraries, etc.)* and the librarian does its job! 

For more customization you can play with how much weight the recommandation system gives to each criteria (for example if you dial the star count slider all the way down, you might find some hidden gems!). This repository contains the whole process from gathering and processing the data, to the web app itself.

**Everything is done locally** *(except the Github API calls needed to get the origianl data of course)*, including the language and embedding models used.


## How to run it

### Setup

**Requirements:** [Ollama](https://github.com/ollama/ollama), more specifically the `embeddinggemma:300m` model to be installed

If you just want to run the app, first **make sure that ollama is running** by running `ollama serve` inside another terminal.

```bash
# Create a virtual environtment
uv venv

# Activate it
source .venv/Scripts/activate # for windows

# Install requirements
uv pip -Ur requirements.txt

# Run the app
streamlit run main.py
```

### Extra setup for data extraction

**Requirements:** [Ollama](https://github.com/ollama/ollama), more specifically these models: `embeddinggemma:300m`, `gemma3:270m` to be installed

The following should be in your `.env` file:

```bash
# Github access token with repo access
GITHUB_TOKEN = "..."
```

After this extra setup, you should be ready to start scarping more of the [awesome lists](#awesome-lists-included) provided, or just setup another list of awesome lists of your preference, by running the `data.ipynb` file.

**Note:** You could always just delete the *checkpoints* folder and start embedding the repos with another model, **everything is setup for interchangeability**. For a more in-depth explanation of how (or why) a language model is used, check out the [data processing](#processing) section.

## How it works

### Data Extraction

The data is extracted via the Github API, then parsed in the following order:

1. [mistletoe](https://github.com/miyuchina/mistletoe) is firstly used to parse the markdown into html
2. [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) is then used in order to extract all the lists (`<ul>` -> `<li>` -> `<a>` -> `href`s )
3. I check for the link to be reffering to another github page, then process it in order to have the following format: `owner/repo`
4. The repo origin is then appended to a list with metadata regarding the awesome list of origin and the subcategory from which it was extracted.

### Processing

The whole processing and storage system is built on top of the **LangChain** framework and **FAISS** vector store provided inside of this library.

I use a **small language model** that gets the first 8000 or so characters of the repo's `readme.md` file. **It summarizes the repo into a paragraph.** At the same time I split the repo readme content into markdown aware chunks for extraction.

The chunks extracted from the readme content go through a process of **context enrichment** by inserting the first sentence of the summary at the begging of each chunk like this:

`[X is an interesing library for Y built in Z] Chunk content`

The **summary chunks** and the **repository readme chunks** are saved inside of **different vector indexes.**

### Weights system

The seach engine is made of three components:

#### 1. Summary Chunks

These are the scores retrieved from the vector index that only contains **summaries of the entire repo.** They use the euclidean distance (L2 distance) for measurement, that are then normalized in a [0;1] range.

#### 2. Repo Chunks

These are the scores retrieved from the vector index that only contains **repo readme content chunks.** The same normailzation technique is used.

#### 3. Star Count

After retrieving the repo and summary chunks i process all the star counts and calculate a **score based on a log scale** in order to follow the scaling laws of the data. 

#### Score Calulation

The scores are then calculated like such:

**FinalScore** = *w1* * **SummaryScore** + *w2* * **RepoScore** + *w3* * **StarScore**

*(with w1, w2 and w3 adding up to 1)*

### Awesome lists included

Here are the lists that are included (you could always just add more!):
* [Awesome-Python](https://github.com/vinta/awesome-python)
* [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning)
* [Awesome JavaScript](https://github.com/sorrycc/awesome-javascript)
* [Awesome Node.js](https://github.com/sindresorhus/awesome-nodejs)
* [Awesome React](https://github.com/enaqx/awesome-react)
* [Awesome Vue.js](https://github.com/vuejs/awesome-vue)
* [Awesome Web Development](https://github.com/nepaul/awesome-web-development)
* [Awesome Security](https://github.com/sbilly/awesome-security)
* [Awesome Big Data](https://github.com/oxnr/awesome-bigdata)
* [Awesome DevOps](https://github.com/wmariuss/awesome-devops)
* [Awesome Robotics](https://github.com/Kiloreux/awesome-robotics)
* [Awesome-Selfhosted](https://github.com/awesome-selfhosted/awesome-selfhosted)
* [Awesome C++](https://github.com/fffaraz/awesome-cpp)
* [Awesome CLI Apps](https://github.com/agarrharr/awesome-cli-apps)

**Note:** The vector index provided only has the scraped data for the **awesome-pyhton** list.

### To do:

- License link fix (in the data extraction process)
- Add more filters that harness the metadata extracted for each chunk

### License

This whole project is under the [MIT License](), so have fun with it!
