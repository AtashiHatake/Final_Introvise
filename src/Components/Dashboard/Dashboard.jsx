import React, { useState, useEffect } from 'react';
import './Dashboard.css';
import { CircularProgressbar } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';
import axios from 'axios';
import jsPDF from 'jspdf';

function Dashboard() {
  const [progressData, setProgressData] = useState({ progress1: 0, progress2: 0, progress3: 0 });

  useEffect(() => {
    const fetchProgressData = async () => {
      try {
        const response = await axios.get('http://localhost:8080/api/progress');
        setProgressData(response.data);
      } catch (error) {
        console.error('Error fetching progress data:', error);
      }
    };

    fetchProgressData();
  }, []);

  const handleDownloadCertificate = async () => {
    try {
      const response = await axios.get('http://localhost:8080/api/certificate');
      const certificateData = response.data;

      const doc = new jsPDF();
      doc.text('Certificate of Achievement', 10, 10);
      doc.text(`Beheviour: ${certificateData.progress1}%`, 10, 20);
      doc.text(`Communication: ${certificateData.progress2}%`, 10, 30);
      doc.text(`Problem Solving: ${certificateData.progress3}%`, 10, 40);
      doc.save('certificate.pdf');
    } catch (error) {
      console.error('Error fetching certificate data:', error);
    }
  };

  const handleDownloadReport = async () => {
    try {
      const response = await axios.get('http://localhost:8080/api/report');
      const reportData = response.data;

      const doc = new jsPDF();
      doc.text('Progress Report', 10, 10);
      doc.text(`Beheviour: ${reportData.progress1}%`, 10, 20);
      doc.text(`Communication: ${reportData.progress2}%`, 10, 30);
      doc.text(`Problem Solving: ${reportData.progress3}%`, 10, 40);
      doc.save('report.pdf');
    } catch (error) {
      console.error('Error fetching report data:', error);
    }
  };

  return (
    <div className="img-dash-container">
      <div className="dashboard-container">
      <h1 className="dashboard-header">Great Job! <br />Interview Done.</h1>
      <h3 className="dashboard-download">Download your analysis report from below.</h3>
      <div className="buttons-container">
        <button className="download-button" onClick={handleDownloadReport}>Download Report</button>
      </div>
    </div>
    </div>
    
  );
}

export default Dashboard;


