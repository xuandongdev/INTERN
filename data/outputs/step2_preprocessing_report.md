
# Step 2: Data Preprocessing Techniques for RAG

## Objective

The objective of Step 2 is to experiment with preprocessing techniques for RAG, including text cleaning, metadata extraction, chunking strategies, and overlap strategies.

## Cleaning

I cleaned the extracted PDF text by removing broken line-break hyphenation, unnecessary newlines, control characters, and repeated spaces.

## Metadata Extraction

I enriched each cleaned page with structured metadata, including source file, source path, page index, page number, document type, character count, word count, and preprocessing stage.

## Chunking Strategies Tested

I tested the following seven chunking strategies:

1. Page-level chunking
2. Fixed-size character chunking
3. Recursive character chunking
4. Sentence-based chunking
5. Semantic chunking
6. Hierarchical chunking
7. LLM-based chunking

## Chunking Summary

           strategy  num_chunks  avg_chars  min_chars  max_chars  avg_words  min_words  max_words
         page_level           6    1614.67       1129       1876     216.50        148        255
         fixed_size          17     640.76         10        800      85.65          2        114
recursive_character          16     623.25        280        778      84.19         38        111
     sentence_based          17     657.35        394        800      88.65         52        111
           semantic          24     402.92          7        975      54.12          1        134
 hierarchical_child          26     386.12         62        498      52.46          8         72
          llm_based          30     167.70         21        488      22.97          4         70

## Pros and Cons

           strategy                                                                          pros                                                             cons                                                                  best_use_case
         page_level                     Preserves full page context and makes page citation easy.           Chunks may be too long and less precise for retrieval.            Short PDF pages or documents where page-level context is important.
         fixed_size          Simple, consistent, and easy to control with chunk size and overlap.                        May cut sentences or ideas in the middle.                                Baseline experiments and simple text documents.
recursive_character Balances chunk size while preserving paragraph and sentence structure better. Still depends on separator choices and chunk size configuration.                 General-purpose RAG preprocessing for PDFs and text documents.
     sentence_based                    Keeps sentences intact and improves readability of chunks. Sentence boundary detection may be imperfect for noisy PDF text.                    Documents with well-formed sentences and clear punctuation.
           semantic    Splits text based on semantic shifts, producing more meaning-aware chunks.             Requires embedding computation and threshold tuning.            Documents with topic changes where semantic coherence is important.
 hierarchical_child     Combines precise child-level retrieval with broader parent-level context.            More complex metadata management and retrieval logic.            Long documents where both precision and broader context are needed.
          llm_based    Can create highly coherent chunks based on meaning and document structure.      Higher cost, higher latency, and less deterministic output. High-value documents where chunk quality is more important than speed or cost.

## Overlap Strategy Experiment

I tested multiple overlap values for recursive character chunking: 0, 80, 120, and 200.

 chunk_size  chunk_overlap  num_chunks  avg_chars  min_chars  max_chars  avg_words  min_words  max_words
        800              0          16     605.50        280        791      81.81         38        111
        800             80          16     617.81        280        778      83.50         38        111
        800            120          16     623.25        280        778      84.19         38        111
        800            200          16     665.56        304        784      90.12         38        111

## Selected Final Pipeline

- Cleaning function: clean_text()
- Metadata extraction: enrich_page_metadata()
- Chunking strategy: recursive_character
- Chunk size: 800
- Chunk overlap: 120

## Reason for Selection

I selected recursive character chunking with chunk_size=800 and chunk_overlap=120 because it provides a good balance between chunk size control, context preservation, stability, and reproducibility.

Compared with fixed-size chunking, it preserves document structure better. Compared with page-level chunking, it avoids overly large chunks. Compared with LLM-based chunking, it is more stable, cost-effective, and easier to reproduce.

## Output

Final processed chunks were saved to:

../data/json/step2_final_processed_chunks.jsonl

## Next Step

Proceed to Step 3: Embedding Methods – Research & Selection.
