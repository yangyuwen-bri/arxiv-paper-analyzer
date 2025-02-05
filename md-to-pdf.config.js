module.exports = {
  stylesheet: [
    'https://cdn.jsdelivr.net/npm/github-markdown-css@5.1.0/github-markdown.min.css'
  ],
  body_class: ['markdown-body'],
  pdf_options: {
    format: 'A4',
    margin: '20mm',
    printBackground: true
  },
  highlight_style: 'github',
  css: `
    body {
      font-family: 'Microsoft YaHei', 'SimSun', Arial, sans-serif;
      line-height: 1.6;
    }
    code {
      font-family: 'Consolas', 'Monaco', monospace;
    }
  `
} 