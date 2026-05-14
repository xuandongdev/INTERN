from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv


def project_root() -> Path:
    root = Path.cwd().resolve()
    if root.name.lower() == "src":
        return root.parent
    return root


def load_config() -> dict[str, str]:
    load_dotenv(find_dotenv(usecwd=True), override=False)

    config = {
        "PROJECT_ID": os.getenv("PROJECT_ID", ""),
        "REGION": os.getenv("REGION", ""),
        "VERTEX_ENDPOINT_RESOURCE_NAME": os.getenv("VERTEX_ENDPOINT_RESOURCE_NAME", ""),
        "VERTEX_ENDPOINT_ID": os.getenv("VERTEX_ENDPOINT_ID", ""),
        "DEPLOYED_INDEX_ID": os.getenv("DEPLOYED_INDEX_ID", ""),
        "EMBEDDING_MODEL_NAME": os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large"),
    }

    missing = [key for key in ("PROJECT_ID", "REGION", "DEPLOYED_INDEX_ID") if not config[key]]
    if missing:
        raise ValueError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Update .env before connecting to Vertex AI Vector Search."
        )

    if not config["VERTEX_ENDPOINT_RESOURCE_NAME"] and not config["VERTEX_ENDPOINT_ID"]:
        raise ValueError(
            "Set VERTEX_ENDPOINT_RESOURCE_NAME or VERTEX_ENDPOINT_ID in .env."
        )

    if not config["VERTEX_ENDPOINT_RESOURCE_NAME"]:
        config["VERTEX_ENDPOINT_RESOURCE_NAME"] = (
            f"projects/{config['PROJECT_ID']}/locations/{config['REGION']}"
            f"/indexEndpoints/{config['VERTEX_ENDPOINT_ID']}"
        )

    return config


def load_chunk_map(root: Path) -> pd.DataFrame:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    data_file = root / "data" / "outputs" / "energy_for_rag.md"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing markdown source: {data_file}")

    raw_markdown = data_file.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n## ", "\n### ", "\n", ". ", " ", ""],
    )
    documents = splitter.create_documents(
        [raw_markdown],
        metadatas=[{"source_file": data_file.name, "source_path": str(data_file)}],
    )

    rows = []
    for index, document in enumerate(documents):
        rows.append(
            {
                "chunk_id": f"energy_chunk_{index:04d}",
                "content": document.page_content,
                "source_file": document.metadata["source_file"],
                "source_path": document.metadata["source_path"],
            }
        )

    return pd.DataFrame(rows)


def extract_neighbor_payload(neighbor, id_to_row: dict[str, dict[str, str]]) -> dict[str, str | float | None]:
    metadata = getattr(neighbor, "embedding_metadata", None) or {}
    local_row = id_to_row.get(neighbor.id, {})
    content = metadata.get("content") or local_row.get("content") or "Content not found."
    source_file = metadata.get("source_file") or local_row.get("source_file")
    source_path = metadata.get("source_path") or local_row.get("source_path")

    return {
        "neighbor_id": neighbor.id,
        "distance": getattr(neighbor, "distance", None),
        "source_file": source_file,
        "source_path": source_path,
        "content": content,
    }


def connect_endpoint(config: dict[str, str]):
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import aiplatform

    try:
        aiplatform.init(project=config["PROJECT_ID"], location=config["REGION"])
        return aiplatform.MatchingEngineIndexEndpoint(
            config["VERTEX_ENDPOINT_RESOURCE_NAME"]
        )
    except DefaultCredentialsError as exc:
        raise RuntimeError(
            "Google Cloud ADC not found. Run:\n"
            "gcloud auth application-default login\n"
            f"gcloud config set project {config['PROJECT_ID']}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Connect to an existing Vertex AI Vector Search deployed index."
    )
    parser.add_argument(
        "--query",
        default="What assumptions are used in the household energy cost analysis?",
        help="Question to send to Vertex AI Vector Search.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of neighbors to retrieve.",
    )
    args = parser.parse_args()

    root = project_root()
    config = load_config()
    chunk_df = load_chunk_map(root)
    endpoint = connect_endpoint(config)

    deployed_indexes = [item.id for item in getattr(endpoint, "deployed_indexes", [])]
    print("Endpoint:", config["VERTEX_ENDPOINT_RESOURCE_NAME"])
    print("Deployed indexes on endpoint:", deployed_indexes or "<none visible>")

    if config["DEPLOYED_INDEX_ID"] not in deployed_indexes:
        print(
            "Warning: DEPLOYED_INDEX_ID is not listed on this endpoint. "
            "If the query fails, re-check the deployed index ID from the console."
        )

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config["EMBEDDING_MODEL_NAME"])
    query_vector = model.encode(
        [f"query: {args.query}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(float).tolist()

    response = endpoint.find_neighbors(
        deployed_index_id=config["DEPLOYED_INDEX_ID"],
        queries=[query_vector],
        num_neighbors=args.top_k,
        return_full_datapoint=True,
    )

    id_to_row = chunk_df.set_index("chunk_id").to_dict(orient="index")
    rows = [extract_neighbor_payload(neighbor, id_to_row) for neighbor in response[0]]

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
