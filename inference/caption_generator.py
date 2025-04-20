import os
import torch
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from utils.image_utils import image_transforms

def beam_search(model, image, vocab, beam_size=3, max_len=20):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0)
        features = model.encoder(image)
        features = model.decoder.img_fc(features)
        sequences = [[list(), 0.0, (None, None)]]

        for _ in range(max_len):
            all_candidates = []
            for seq, score, (h, c) in sequences:
                if len(seq) > 0 and seq[-1] == vocab.word2idx['<eos>']:
                    all_candidates.append((seq, score, (h, c)))
                    continue
                input_seq = torch.tensor([seq[-1]] if seq else [vocab.word2idx['<sos>']])
                embedded = model.decoder.embed(input_seq).unsqueeze(1)

                if len(seq) == 0:
                    embedded = features.unsqueeze(1)
                output, (h, c) = model.decoder.lstm(embedded, (h, c) if h is not None else None)
                scores = model.decoder.linear(output.squeeze(1))
                log_probs = torch.log_softmax(scores, dim=1)
                top_k_probs, top_k_idx = torch.topk(log_probs, beam_size)
                for i in range(beam_size):
                    candidate = seq + [top_k_idx[0][i].item()]
                    candidate_score = score + top_k_probs[0][i].item()
                    all_candidates.append((candidate, candidate_score, (h, c)))

            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
            sequences = ordered[:beam_size]

        final_seq = sequences[0][0]
        caption = [vocab.idx2word[idx] for idx in final_seq if idx not in [vocab.word2idx['<sos>'], vocab.word2idx['<eos>'], vocab.word2idx['<pad>']]]
        return ' '.join(caption)
    

def generate_caption(model, image_input, vocab, beam_size=3, max_len=20):
    model.eval()

    try:
        if isinstance(image_input, str):
            if image_input.startswith("http://") or image_input.startswith("https://"):
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0",
                    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                    "Referer": image_input.split("/")[2],
                }
                response = requests.get(image_input, headers=headers, stream=True, timeout=10)

                if response.status_code == 403:
                    return "Access denied: The image URL is forbidden (403). Try a different source."
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "")
                if "image" not in content_type:
                    return "The URL does not point to a valid image. Please check the link."

                try:
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                except UnidentifiedImageError:
                    return "Couldn't identify image format. Try another image link."

            elif os.path.exists(image_input):
                try:
                    image = Image.open(image_input).convert("RGB")
                except UnidentifiedImageError:
                    return "Failed to load local image. Make sure it's a supported format."

            else:
                return "The provided path or URL is invalid."

            transform = image_transforms()
            image = transform(image).unsqueeze(0)

        elif isinstance(image_input, torch.Tensor):
            image = image_input

        else:
            return "Unsupported image input type."

        caption = beam_search(model, image, vocab, beam_size, max_len)
        return caption

    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"

    except Exception as e:
        return f"Unexpected error: {str(e)}"
