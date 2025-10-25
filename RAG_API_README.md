to run the API 

```
cd DeepSeek-OCR-master-DeepSeek-OCR-vllm
python api.py
```

example of a prompt fed to chatGPT to extract paper information:

```
Pasted below is a MultiMarkdown file of an academic paper that has been extracted through an OCR tool.

From it, you are to extract the following fields and return the response as a JSON.
- title
- abstract
- authors (list of strings)
- keywords (list of strings)
- images (list of dictionaries, each having an identifier (ex. Figure 1), path, caption, and list of sections in which it is referenced)
- tables (list of dictionaries, each having an identifier (ex. Table 1), value (raw markdown text), caption, and list of sections in which it is referenced)

--- MULTIMARKDOWN FILE START ---

--- MULTIMARKDOWN FILE END ---

Parse the file and output the JSON only, do not output any additional text
```