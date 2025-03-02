import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import './hide.css'
import { Dashboard } from '@mui/icons-material'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
    <Dashboard />
  </StrictMode>,
)
