import os

# Configure variables
# This is the passage that is being queried to find the answer
initial_passage = "/workspace/TensorRT/demo/BERT/docs/Medical_Records/22532_Short_Document_With_Answers.txt"

# This is the question that is being asked. The answer is searched for in initial_passage.
question = "How many patients experienced recurrence at 12 years of age?"

# BERT Engine to use
BERT_engine = "bert_base_384.engine"

# Vocabulary to use
vocabulary = "/workspace/models/fine-tuned/bert_tf_v2_base_fp16_384_v2/vocab.txt"

# Batch size for inference (default=1
batch_size_for_inference = 1

# Sequence length for query
sequence_length = 5000

# answer_and_probability is used to hold the answers that BERT returns
answer_and_probability = []



# Read and prepare the initial_passage
# Read the passage into a variable
f=open(initial_passage, "r")
contents =f.read()
# Escape parentheses as these cause issues
contents = contents.replace("(", "\(")
contents = contents.replace(")", "\)")

# Split contents into a list with new line as the separator.  This will break the initial_passage
# into distinct paragraphs.  These will most likely be shorter than the maximum length that can
# be queried.
# TODO:  This should be checked and if longer than the max length, break the paragraph into pieces
paragraphs = contents.split('\n')

# Remove any list items that are just a new line with no text.  These are simply blank lines
# between paragraphs.  They will show up as ''
for i in range((len(paragraphs)-1), 0, -1):
	if paragraphs[i] == '':
		del paragraphs[i] 



# Loop through the items in paragraphs and use each as the passage for the BERT query
for i in range(0, len(paragraphs)):
	# Print the number of the paragraph being read.  This is just to keep track of progress.
	print("\n\nReading paragraph #" + str(i + 1) + " of " + str(len(paragraphs)))

	# Create the command
	bert_command = "python python/bert_inference.py -e " + BERT_engine + " -p " + paragraphs[i] + " -q " + question + " -v " + vocabulary + " -b " + str(batch_size_for_inference) + " -s " + str(sequence_length) + " | grep -e Answer -e With"
	# Run the command and save the output to bert_output
	bert_output = os.popen(bert_command).read()
	
	# Check the output that was saved in bert_output.  If the answer is blank, no need to actually
	# save it.  If the answer is not blank, append the answer to answer_and_probability
	# Also, remove the trailing '\n' because this causes issues when parsing the answers below
	if "Answer: ''" not in bert_output and bert_output != '':
		answer_and_probability.append(bert_output.rstrip("\n"))



# Get answer with highest probability and save that as the final answer.  If no answers were found
# then answer_and_probability will have a length of zero and a message that no answer was found
# should be displayed
prob = 0.0
answer = ''

if len(answer_and_probability) > 0:

	# Get the answer with the highest probability
	for i in range(0, len(answer_and_probability) - 1):

		# Get the answer with the highest probability	
		if float(answer_and_probability[i].split(" ")[3]) >= prob:
			prob = float(answer_and_probability[i].split(" ")[3])
			answer = answer_and_probability[i].split(" ")[1].rstrip("\nWith")

	# Print final output
	print("\n\nThe question was \"" + question + "\"")
	print("The best answer is {0} with a probability of {1:.2f}%".format(answer, prob))
else:
		# Print final output
	print("\n\nThe question was \"" + question + "\", but no answer was found.")

