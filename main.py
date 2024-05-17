#!/usr/bin/python3

from absl import flags, app
from transformers import AutoTokenizer
from huggingface_hub import login
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts.prompt import PromptTemplate
import gradio as gr

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('host', default = '0.0.0.0', help = 'host address')
  flags.DEFINE_integer('port', default = 8081, help = 'port number')

def CodeLlama(locally = False):
  login(token = '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')
  tokenizer = AutoTokenizer.from_pretrained('meta-llama/CodeLlama-7b-Instruct-hf')
  llm = HuggingFacePipeline.from_model_id(
    model_id = 'meta-llama/CodeLlama-7b-Instruct-hf',
    task = 'text-generation',
    device = 0,
    pipeline_kwargs = {
      "max_length": 16384,
      "do_sample": False,
      "temperature": 0.8,
      "top_p": 0.8,
      "use_cache": True,
      "return_full_text": False
    }
  )
  return tokenizer, llm

def main(unused_argv):
  tokenizer, llm = CodeLlama(True)
  def query(question, history):
    messages = list()
    for q,a in history[-5:]:
      messages.append({'role':'user', 'content':q})
      messages.append({'role':'assistant', 'content':a})
    messages.append({'role':'user', 'content': '{question}'})
    prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    template = PromptTemplate(template = prompt, input_variables = ['question'])
    chain = template | llm
    answer = chain.invoke({'question': question})
    history.append((question, answer))
    return "", history
  block = gr.Blocks()
  with block as demo:
    with gr.Row(equal_height = True):
      with gr.Column(scale = 15):
        gr.Markdown("<h1><center>文献问答系统</center></h1>")
    with gr.Row():
      with gr.Column(scale = 4):
        chatbot = gr.Chatbot(height = 450, show_copy_button = True)
        msg = gr.Textbox(label = "需要问什么？")
        with gr.Row():
          submit_btn = gr.Button("发送")
          clear_btn = gr.ClearButton(components = [chatbot], value = "清空问题")
      submit_btn.click(query, inputs = [msg, chatbot], outputs = [msg, chatbot])
  gr.close_all()
  demo.launch(server_name = FLAGS.host, server_port = FLAGS.port)

if __name__ == "__main__":
  add_options()
  app.run(main)

