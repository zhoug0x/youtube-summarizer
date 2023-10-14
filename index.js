import fs from "fs";
import { loadSummarizationChain } from "langchain/chains";
import { YoutubeLoader } from "langchain/document_loaders/web/youtube";
import { Ollama } from "langchain/llms/ollama";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import path from "path";

//--------------------------- SETTINGS ----------------------------------------

const VIDEO_URL = "https://www.youtube.com/watch?v=5-TgqZ8nado";

const DO_FETCH = true;
const DO_SPLIT = true;
const DO_SUMMARIZE = true;

const CHUNK_SIZE = 500;
const CHUNK_OVERLAP = 10;
const SUMMARIZATION_TYPE = "refine"; // 'stuff', 'refine' or 'map_reduce', different ones == different results
const MODEL_TYPE = "mistral-openorca"; // instal the model locally via ollama: https://ollama.ai

//-----------------------------------------------------------------------------

const __filename = new URL(import.meta.url).pathname;
const __dirname = path.dirname(__filename);

// load a video's transcription & metadata from youtube w/ langchain
const loadYoutubeData = async (url) => {
  const loader = YoutubeLoader.createFromUrl(url, {
    language: "en",
    addVideoInfo: true,
  });
  const [doc] = await loader.load();
  return doc;
};

//-----------------------------------------------------------------------------

const main = async () => {
  let doc;
  let docs;

  console.log("\n\n🏁 starting job...\n");

  if (DO_FETCH) {
    console.log("\n⤵️ fetching youtube data...\n");
    doc = await loadYoutubeData(VIDEO_URL);
    fs.writeFileSync(
      path.join(__dirname, "doc-single.json"),
      JSON.stringify(doc, null, 2)
    );
  } else {
    console.log("\n⤴️ loading youtube data from file...\n");
    doc = JSON.parse(fs.readFileSync(path.join(__dirname, "doc-single.json")));
  }

  if (DO_SPLIT) {
    // split up the transcript into chunks
    console.log("\n🪵🪓 splitting data...\n");
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: CHUNK_SIZE,
      chunkOverlap: CHUNK_OVERLAP,
    });
    docs = await splitter.splitDocuments([doc]);

    // write the split results to a file with multiple document entries
    fs.writeFileSync(
      path.join(__dirname, "doc-split.json"),
      JSON.stringify(docs, null, 2)
    );
  } else {
    console.log("\n⤴️ loading split data from file...\n");
    docs = JSON.parse(fs.readFileSync(path.join(__dirname, "doc-split.json")));
  }

  if (DO_SUMMARIZE) {
    console.log("\n🤖 summarizing...\n");

    // summarize the docs & save result summarization to a file
    const model = new Ollama({
      baseUrl: "http://localhost:11434", // ollama must be running: `ollama serve`
      model: MODEL_TYPE,
    });

    const chain = loadSummarizationChain(model, { type: SUMMARIZATION_TYPE });
    const result = await chain.call({ input_documents: docs });

    fs.writeFileSync(
      path.join(__dirname, "doc-summarized.txt"),
      JSON.stringify(result.text, null, 2)
    );
  }

  console.log("\n✅ complete!\n\n");
};

main();
