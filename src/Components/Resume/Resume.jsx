import React, { useState } from "react";
import "./resume.css";

const jobRoles = ["Software Developer", "Data Analyst", "AI/ML Engineer", "DevOps Engineer", "Cloud Engineer"];
const jobTitles = ["Junior", "Senior"];

const skillsData = {
    "Software Developer": {
        Junior: ["HTML", "CSS", "JS", "Node.js", "Python", "SQL", "Git", "Debugging"],
        Senior: ["System Design", "Microservices", "Cloud (AWS/GCP)", "CI/CD", "Security"],
    },
    "Data Analyst": {
        Junior: ["SQL", "Excel", "Python (Pandas, Matplotlib)", "Data Visualization"],
        Senior: ["Big Data (Spark, Hadoop)", "Predictive Analytics", "BI Tools", "Cloud Data Services"],
    },
    "AI/ML Engineer": {
        Junior: ["Python", "ML Libraries (Scikit-learn, TensorFlow)", "Data Preprocessing"],
        Senior: ["Deep Learning", "Model Deployment", "MLOps", "Hyperparameter Tuning"],
    },
    "DevOps Engineer": {
        Junior: ["Linux", "Git", "Docker", "CI/CD"],
        Senior: ["Kubernetes", "Terraform", "Security", "Monitoring (Prometheus)"],
    },
    "Cloud Engineer": {
        Junior: ["Cloud Basics (AWS, GCP)", "Networking", "Linux", "Terraform"],
        Senior: ["Cloud Architecture", "Serverless", "Security (IAM, VPC)", "Scaling"],
    },
};

const JobSkillsSelector = ({ setJobTitle, setRequiredSkills }) => {
    const [selectedRole, setSelectedRole] = useState("");
    const [selectedTitle, setSelectedTitle] = useState("");
    const [availableSkills, setAvailableSkills] = useState([]);
    const [selectedSkills, setSelectedSkills] = useState([]);

    const handleRoleChange = (e) => {
        const role = e.target.value;
        setSelectedRole(role);
        setJobTitle(role);
        setSelectedTitle("");
        setAvailableSkills([]);
        setSelectedSkills([]);
        setRequiredSkills([]);
    };

    const handleTitleChange = (e) => {
        const title = e.target.value;
        setSelectedTitle(title);
        const skills = skillsData[selectedRole][title] || [];
        setAvailableSkills(skills);
        setSelectedSkills([]);
        setRequiredSkills([]);
    };

    const handleSkillChange = (skill) => {
        let updatedSkills = [...selectedSkills];
        if (updatedSkills.includes(skill)) {
            updatedSkills = updatedSkills.filter(s => s !== skill);
        } else {
            updatedSkills.push(skill);
        }
        setSelectedSkills(updatedSkills);
        setRequiredSkills(updatedSkills);
    };

    return (
        <div className="box">
            <img className="img-r" src="./public/resume_img.png" alt="Introvise/public/resume_img.png" />
            <div className="job-skills-selector">
                {/* Job Role Dropdown */}
                <select onChange={handleRoleChange} className="dropdown">
                    <option value="" disabled selected>Select Job Role</option>
                    {jobRoles.map((role) => (
                        <option key={role} value={role}>{role}</option>
                    ))}
                </select>

                {/* Job Title Dropdown */}
                <select onChange={handleTitleChange} className="dropdown2" value={selectedTitle} disabled={!selectedRole}>
                    <option value="" disabled>Select Job Title</option>
                    {jobTitles.map((title) => (
                        <option key={title} value={title}>{title}</option>
                    ))}
                </select>

                {/* Required Skills with Checkboxes */}
                {selectedRole && selectedTitle && (
                    <div className="skills-list">
                        <h3>Required Skills:</h3>
                        {availableSkills.map(skill => (
                            <label key={skill} className="skill-item">
                                <input 
                                    type="checkbox" 
                                    checked={selectedSkills.includes(skill)}
                                    onChange={() => handleSkillChange(skill)}
                                /> 
                                {skill}
                            </label>
                        ))}
                    </div>
                )}
            </div>
            <button className="btn-r">Submit</button>
        </div>
    );
};

export default JobSkillsSelector;
