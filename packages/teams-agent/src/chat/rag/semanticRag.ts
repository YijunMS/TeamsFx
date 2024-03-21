//import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
//import { MemoryVectorStore } from "langchain/vectorstores/memory";
//import { similarity } from "ml-distance";
import { _importDynamic } from "./ragUtil";

async function embeddingsHelloWorld() {
  let mod1 = await _importDynamic('@langchain/community/embeddings/hf_transformers');
  const model = new mod1.HuggingFaceTransformersEmbeddings({
    modelName: "Xenova/all-MiniLM-L6-v2",
  });

  /* Embed queries */
  const res = await model.embedQuery(
    "What would be a good company name for a company that makes colorful socks?"
  );
  console.log({ res });
  /* Embed documents */
  //const documentRes = await model.embedDocuments(docs);
  //console.log({ documentRes });

  //const vectorStore = await MemoryVectorStore.fromTexts(
  //  docs,
  //  metadata,
  //  model,
  //  { similarity: similarity.cosine }
  //);
  //console.log(await vectorStore.similaritySearch("change the text of a comment", 5));
  //console.log(await vectorStore.asRetriever().getRelevantDocuments("resolve the first comment"));
}

export async function initModelByDocAndMetadata(docs: string[], metadata: object[]) {
  let mod1 = await _importDynamic('@langchain/community/embeddings/hf_transformers');
  let mod2 = await _importDynamic('langchain/vectorstores/memory');
  let mod3 = await _importDynamic('ml-distance');
  let faiss = await _importDynamic('@langchain/community/vectorstores/faiss');
  const model = new mod1.HuggingFaceTransformersEmbeddings({
    modelName: "Xenova/all-MiniLM-L6-v2",
  });
  // const vectorStore = mod2.MemoryVectorStore.fromTexts(
  //   docs,
  //   metadata,
  //   model,
  //   { similarity: mod3.similarity.cosine }
  // );
  // return vectorStore;

  {
    const vectorStore = await faiss.FaissStore.fromTexts(
      docs,
      metadata,
      model,
      { similarity: mod3.similarity.cosine }
    )
    await vectorStore.save("./vector.txt");
    return vectorStore
  }

  // {
  //   const vectorStore = await faiss.FaissStore.load("./vector.txt",
  //     model,
  //     { similarity: mod3.similarity.cosine }
  //   );
  //   return vectorStore
  // }
}
