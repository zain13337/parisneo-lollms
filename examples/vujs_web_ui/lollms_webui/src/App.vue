<template>
  <div class="bg-gray-900 text-white min-h-screen p-4">
    <h1 class="text-3xl font-bold mb-4">Lord Of Large Language Models</h1>
    <div class="mb-4">
      <h2 class="text-xl font-bold">Select Binding</h2>
      <select v-model="selectedBinding" @change="selectBinding" class="p-2 bg-gray-800 text-white">
        <option v-for="binding in bindings" :key="binding.name" :value="binding.name">{{ binding.name }}</option>
      </select>
    </div>
    <div v-if="selectedBinding" class="mb-4">
      <h2 class="text-xl font-bold">Select Model</h2>
      <select v-model="selectedModel" @change="selectModel" class="p-2 bg-gray-800 text-white">
        <option v-for="model in models" :key="model.title" :value="model.title">{{ model.title }}</option>
      </select>
    </div>
    <div v-if="selectedModel" class="mb-4">
      <h2 class="text-xl font-bold">Select Personality</h2>
      <select v-model="selectedPersonality" @change="selectPersonality" class="p-2 bg-gray-800 text-white">
        <option v-for="personality in personalities" :key="personality.name" :value="personality.name">{{ personality.name }}</option>
      </select>
    </div>
    <div>
      <h2 class="text-xl font-bold">Chat</h2>
      <div class="mb-4">
        <div v-for="message in chatMessages" :key="message.id" class="text-white">
          <strong>{{ message.sender }}:</strong> {{ message.text }}
        </div>
      </div>
      <div class="flex">
        <input type="text" v-model="inputMessage" @keydown.enter="sendMessage" placeholder="Type your message" class="p-2 flex-grow bg-gray-800 text-white mr-2">
        <button @click="sendMessage" class="p-2 bg-blue-500 text-white">Send</button>
      </div>
    </div>
  </div>
</template>

<style src="./assets/css/app.css"></style>
<script>
import io from 'socket.io-client';
// Import Tailwind CSS styles
import 'tailwindcss/tailwind.css';

export default {
  data() {
    return {
      socket: null,
      bindings: [],
      models: [],
      personalities: [],
      selectedBinding: '',
      selectedModel: '',
      selectedPersonality: '',
      chatMessages: [],
      inputMessage: '',
    };
  },
  created() {
    this.socket = io('http://localhost:9600');
    this.socket.on('connect', () => {
      console.log('Connected to server');
      this.socket.emit('list_available_bindings');
      this.socket.emit('list_available_models');
      this.socket.emit('list_available_personalities');
    });
    // Handle the event emitted when the select_binding is sent
    this.socket.on('select_binding', (data) => {
      console.log('Received:', data);
      if(data["success"]){
        console.log('Binding selected:', data);
        this.socket.emit('list_available_models');
      }
      // You can perform any additional actions or update data properties as needed
    });
    // Handle the event emitted when the select_binding is sent
    this.socket.on('select_model', (data) => {
      console.log('Received:', data);
      if(data["success"]){
        console.log('Model selected:', data);
      }
      // You can perform any additional actions or update data properties as needed
    });

    this.socket.on('bindings_list', (bindings) => {
      this.bindings = bindings["bindings"];
      console.log(this.bindings)
    });

    this.socket.on('available_models_list', (models) => {
      if(models["success"]){
        this.models = models["available_models"];
      }
      console.log(this.models)
    });

    this.socket.on('personalities_list', (personalities) => {
      this.personalities = personalities;
    });
    
    this.socket.on('text_chunk', (message) => {
      this.chatMessages.push(message.chunk);
    });
  },
  methods: {
    selectBinding() {
      this.socket.emit('select_binding', { binding_name: this.selectedBinding });
    },
    selectModel() {
      this.socket.emit('select_model', { model_name: this.selectedModel });
    },
    selectPersonality() {
      this.socket.emit('activate_personality', { personality_name: this.selectedPersonality });
    },
    sendMessage() {
      const message = {
        text: this.inputMessage,
        sender: 'User',
      };
      this.chatMessages.push(message);
      this.socket.emit('generate_text', {prompt:message.text, personality:0});
      this.inputMessage = '';
    },
  },
};
</script>
