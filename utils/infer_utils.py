import os
import json
import argparse
from typing import List, Dict
from dataclasses import dataclass
from openai import OpenAI
from tqdm import tqdm

@dataclass
class Config:
	statutes_path: str
	cases_path: str
	model_name: str
	api_base_url: str
	api_key: str
	output_path: str
	num_waits: int
	token_budget: int
	debug: bool

def parse_args() -> Config:
	parser = argparse.ArgumentParser(description='Process tax cases using LLM inference')
	parser.add_argument('--statutes-path', required=True, help='Path to statutes directory')
	parser.add_argument('--cases-path', required=True, help='Path to cases directory')
	parser.add_argument('--model-name', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
					  help='Name of the model to use')
	parser.add_argument('--api-base-url', default='http://localhost:9009/v1',
					  help='Base URL for the API')
	parser.add_argument('--api-key', default='token-abc123',
					  help='API key for authentication')
	parser.add_argument('--output-path', default='out/output.json',
					  help='Path to save output JSON')
	parser.add_argument('--num-waits', type=int, default=0, help='Number of waits (default: 0)')
	parser.add_argument('--token-budget', type=int, default=8000, help='Token budget for model inference (default: 8000)')
	parser.add_argument('--debug', default=False, action='store_true', help='Only run on a two examples')

	args = parser.parse_args()
	return Config(
		statutes_path=args.statutes_path,
		cases_path=args.cases_path,
		model_name=args.model_name,
		api_base_url=args.api_base_url,
		api_key=args.api_key,
		output_path=args.output_path,
		num_waits=args.num_waits,
		token_budget=args.token_budget,
		debug=args.debug
	)

def load_statutes(statutes_path: str) -> str:
	"""Load and combine all statute files from the given directory."""
	file_texts = []
	try:
		for filename in os.listdir(statutes_path):
			file_path = os.path.join(statutes_path, filename)
			if filename[0] == ".":
				continue
			if os.path.isfile(file_path):
				with open(file_path, 'r', encoding='utf-8') as file:
					file_texts.append(file.read())
		return "\n\n\n\n".join(file_texts)
	except Exception as e:
		raise RuntimeError(f"Error loading statutes: {str(e)}")

def load_cases(cases_path: str) -> List[Dict]:
	"""Load and parse tax cases from the given directory."""
	cases = []
	try:
		case_files = [f for f in os.listdir(cases_path) if f.startswith("tax")]
		for case_file in case_files:
			with open(os.path.join(cases_path, case_file), 'r', encoding='utf-8') as file:
				lines = [line.strip() for line in file if line.startswith("%")]
				case_dict = {}
				for i, line in enumerate(lines):
					if line == "% Text":
						case_dict['text'] = lines[i+1].strip("% ")
					if line == "% Question":
						question, answer = lines[i+1].strip("% ").split(" $")
						case_dict['question'] = question
						case_dict['label'] = answer
				cases.append(case_dict)
		return cases
	except Exception as e:
		raise RuntimeError(f"Error loading cases: {str(e)}")

def compose_prompt(statutes: str, case: Dict) -> str:
	"""Create a formatted prompt for the model."""
	return (f"Statutes:\n{statutes}\n\n"
			f"Case: {case['text']}\n\n"
			f"Question: {case['question']}\n\n"
			"Answer the question based on the case and statutes above. "
			"Your answer should be a dollar figure. "
			"Indicate your answer using \\boxed{}")

def run_inference(cases: List[Dict], statutes: str, config: Config) -> List[Dict]:
	"""Run model inference on all cases."""
	client = OpenAI(base_url=config.api_base_url, api_key=config.api_key)

	for case in tqdm(cases, desc="Processing cases"):
		prompt = compose_prompt(statutes, case)
		try:
			response = client.chat.completions.create(
				model=config.model_name,
				messages=[{"role": "user", "content": prompt}]
			)

			case['prompt'] = prompt
			case['answer'] = response.choices[0].message.content
		except Exception as e:
			print(f"Error processing case: {str(e)}")
			case['answer'] = "ERROR"
	
	return cases


def run_inference_forced(cases: List[Dict], statutes: str, config: Config) -> List[Dict]:
	"""Run model inference on all cases."""
	client = OpenAI(base_url=config.api_base_url, api_key=config.api_key)
	for case in tqdm(cases, desc="Processing cases"):
		prompt = compose_prompt(statutes, case)
		try:
			model_prompt = f"""<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜><think>\n"""

			# print(prompt.format(prompt="hello world"))
			stop_token = "</think>"
			total_tokens_used = 0
			total_generated_text = ""
			for i in range(config.num_waits + 1):
				response = client.completions.create(
					model=config.model_name,
					prompt=model_prompt + total_generated_text,
					max_tokens=config.token_budget - total_tokens_used,
					stop=stop_token #make argument
				)
				reason = response.choices[0].stop_reason
				total_tokens_used += response.usage.completion_tokens
				total_generated_text += response.choices[0].text + "\nWait"
				if total_tokens_used >= config.token_budget or reason != stop_token:
					break
				print(f"tried to stop after {total_tokens_used} tokens")
			print(f"actually stopped after {total_tokens_used} tokens")
			# response = client.completions.create(
			#	 model=config.model_name,
			#	 prompt=model_prompt + total_generated_text,
			#	 max_tokens=config.token_budget - total_tokens_used,
			#	 stop=stop_token #make argument
			# )
			case['prompt'] = prompt
			case['answer'] = total_generated_text
		except Exception as e:
			print(f"Error processing case: {str(e)}")
			case['answer'] = "ERROR"
	
	return cases

def save_results(cases: List[Dict], output_path: str):
	"""Save results to JSON file."""
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	try:
		with open(output_path, "w") as f:
			json.dump(cases, f, indent=2)
	except Exception as e:
		raise RuntimeError(f"Error saving results: {str(e)}")


def format_chess_prompt(fen, move):
	assistant = "white" if fen.split(" ")[1] == "b" else "black"
	user = "black" if assistant == "white" else "white"

	return ("Forsyth-Edwards Notation (FEN) is a standard notation for describing a particular board position of a chess game. The purpose of FEN is to provide all the necessary information to restart a game from a particular position. "
			"A FEN record defines a particular game position, all in one text line and using only the ASCII character set.\n\n"
			"A FEN record contains six fields, each separated by a space. The fields are as follows:\n\n"
			"1. Piece placement data: Each rank is described, starting with rank 8 and ending with rank 1, with a '/' between each one; within each rank, the contents of the squares are described in order from the a-file to the h-file. Each piece is identified by a single letter taken from the standard English names in algebraic notation (pawn = 'P', knight = 'N', bishop = 'B', rook = 'R', queen = 'Q' and king = 'K'). White pieces are designated using uppercase letters ('PNBRQK'), while black pieces use lowercase letters ('pnbrqk'). A set of one or more consecutive empty squares within a rank is denoted by a digit from '1' to '8', corresponding to the number of squares.\n"
			"2. Active color: 'w' means that White is to move; 'b' means that Black is to move.\n"
			"3. Castling availability: If neither side has the ability to castle, this field uses the character '-'. Otherwise, this field contains one or more letters: 'K' if White can castle kingside, 'Q' if White can castle queenside, 'k' if Black can castle kingside, and 'q' if Black can castle queenside. A situation that temporarily prevents castling does not prevent the use of this notation.\n"
			"4. En passant target square: This is a square over which a pawn has just passed while moving two squares; it is given in algebraic notation. If there is no en passant target square, this field uses the character '-'. This is recorded regardless of whether there is a pawn in position to capture en passant. An updated version of the spec has since made it so the target square is recorded only if a legal en passant capture is possible, but the old version of the standard is the one most commonly used.\n"
			"5. Halfmove clock: The number of halfmoves since the last capture or pawn advance, used for the fifty-move rule.\n"
			"6. Fullmove number: The number of the full moves. It starts at 1 and is incremented after Black's move.\n\n"
			"Below are two examples of FEN notation. Here is the FEN for the starting position: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n"
			"And after the move 1. e4: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\n\n"
			f"The current board position is given by {fen}."
			f"In this position, {user} just made the move {move}, indicating that the piece on {move[:2]} moved to {move[2:]}. What is the best response as {assistant}?"
			"Indicate your response in the same notation with \\boxed{}.")

def parse_fen(fen):
	pos = fen.split(" ")[0]
	
	p2s = defaultdict(lambda: [])
	s2p = defaultdict(lambda: "")

	for r, rank in enumerate(pos.split("/")):
		f = 0
		for piece in rank:
			if piece.isdigit():
				f += int(piece)
			else:
				square = chr(97+f) + str(8-r)
				s2p[square] = piece
				p2s[piece].append(square)

	return p2s, s2p

def explain_fen(fen, move):
	pos = fen.split(" ")[0]
	res = []
	i = 8

	if move is not None:
		start, end = move[:2], move[2:]
		spiece = None
		epiece = None

	piece_dict = {
		"q": "queen",
		"k": "king",
		"p": "pawn",
		"b": "bishop",
		"r": "rook",
		"n": "knight",
	}

	res.append("Let's understand the position by considering each rank.")
	for rank in pos.split("/"):
		res.append(f"Rank {i} is given by {rank}.")
		f = 0
		for piece in rank:
			if piece.isdigit():
				for j in range(int(piece)):
					res.append(chr(97+f+j)+str(i)+" is empty.")
				f += int(piece)
			else:
				color = "white" if piece.isupper() else "black"
				piece = piece_dict[piece.lower()]
				res.append(f"There is a {color} {piece} on " + chr(97+f) + str(i) + ".")
				f += 1

				if start in res[-1]:
					spiece = [color, piece]
				if end in res[-1]:
					epiece = [color, piece]

		assert f == 8
		i -= 1
	
	if move is not None:
		s = f"{spiece[0].capitalize()} just moved their {spiece[1]} from {start}"	 
		if epiece is None:
			s += f" to {end}."
		else:
			s += f", taking the {epiece[0]} {epiece[1]} on {end}."

		res.append(s)
		
	return " ".join(res)

def main():
	config = parse_args()
	# print('debug:', config.debug, type(config.debug))

	try:
		# Load data
		statutes = load_statutes(config.statutes_path)
		cases = load_cases(config.cases_path)

		if config.debug:
			cases = cases[:2]

		# Run inference
		cases = run_inference(cases, statutes, config)
		#cases = run_inference_forced(cases, statutes, config)

		
		# Save results
		save_results(cases, config.output_path)
		
		print(f"Processing complete. Results saved to {config.output_path}")
		
	except Exception as e:
		print(f"Error during execution: {str(e)}")
		exit(1)

if __name__ == "__main__":
	main()
