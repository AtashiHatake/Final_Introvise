// eslint-disable-next-line no-unused-vars
import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import ResponsiveAppBar from './Components/Navbar/Navbar';
import Login from './Components/Login/Login';
import Resume from './Components/Resume/Resume';
import Facecam from './Components/Facecam/Facecam';
import Dashboard from './Components/Dashboard/Dashboard';
import Landing from './Components/Landing/Landing';


const App = () => {
  const location = useLocation();
  const showNavbar = location.pathname !== '/';

  return (
    <div>
      {showNavbar && <ResponsiveAppBar />}
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/resume" element={<Resume />} />
        <Route path="/facecam" element={<Facecam />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/landing" element={<Landing/>} />
      </Routes>
    </div>
  );
};

const AppWrapper = () => (
  <Router>
    <App />
  </Router>
);

export default AppWrapper;