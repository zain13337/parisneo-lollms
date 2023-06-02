const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true,
  css: {
    loaderOptions: {
      css: {
        // Import the tailwind.css file
        import: 'assets/css/tailwind.css'
      }
    }
  }
})
