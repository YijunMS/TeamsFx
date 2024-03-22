//import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
//import { MemoryVectorStore } from "langchain/vectorstores/memory";
//import { similarity } from "ml-distance";
import { prepareDocs } from "./rag";
import { _importDynamic } from "./ragUtil";

export class SemanticRag {
  private static instance: SemanticRag | null = null;
  private vectorStore: any = null;

  private hf_transformers: any = null;
  private ml_distance: any = null;
  private faiss: any = null;


  constructor() {
  }

  public async init() {
    this.faiss = await _importDynamic('@langchain/community/vectorstores/faiss');
    this.ml_distance = await _importDynamic('ml-distance');
    this.hf_transformers = await _importDynamic('@langchain/community/embeddings/hf_transformers');
    const [docs, metadata] = prepareDocs();
    this.vectorStore = await this.initModelByDocAndMetadata(Array.from(metadata.keys()), Array.from(metadata.values()));
  }

  public static async getInstance(): Promise<SemanticRag> {
    if (!SemanticRag.instance) {
      SemanticRag.instance = new SemanticRag();
      await SemanticRag.instance.init();
    }
    return SemanticRag.instance;
  }

  public async getSimilarity(text: string, n: number): Promise<any> {
    const res = await this.vectorStore.similaritySearch(text, n);
    return res;
  }

  private async initModelByDocAndMetadata(docs: string[], metadata: object[]) {
    const model = new this.hf_transformers.HuggingFaceTransformersEmbeddings({
      modelName: "Xenova/all-MiniLM-L6-v2",
    });
    const directory = "./sematicRagVector"
    const loadFromFile = false;

    let vectorStore = await this.faiss.FaissStore.load(
      directory,
      model,
      { similarity: this.ml_distance.similarity.cosine }
    );
    if (!vectorStore) {
      vectorStore = await this.faiss.FaissStore.fromTexts(
        docs,
        metadata,
        model,
        { similarity: this.ml_distance.similarity.cosine }
      )
      await vectorStore.save(directory);
    }
    return vectorStore;
  }

}

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
