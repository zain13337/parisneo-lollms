import { createRouter, createWebHistory } from 'vue-router'
import PlayGroundView from '../views/PlayGroundView.vue'


const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'playground',
      component: PlayGroundView
    },

  ],
 
})


export default router
