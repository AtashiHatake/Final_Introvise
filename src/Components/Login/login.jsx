import React, { useState, useEffect } from "react";
import { GoogleOAuthProvider, GoogleLogin } from "@react-oauth/google";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "./login.css";
import Slider from "react-slick";
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";

const settings = {
  dots: false,
  infinite: true,
  speed: 500,
  slidesToShow: 1,
  slidesToScroll: 1,
  autoplay: true,          
  autoplaySpeed: 2500,     
  pauseOnHover: false,  // scrolls even when you hover
};

const clientId = "315124824926-eojplkm74o08v3qrsqjuudqbkfctmnhl.apps.googleusercontent.com";

function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [user, setUser] = useState(null);
  const navigate = useNavigate();

  // Handle Manual Sign-In
  const handleSignIn = async () => {
    if (!email || !password) {
      alert("Please enter both email and password!");
      return;
    }

    try {
      const res = await axios.post("http://localhost:8080/api/users/login1", {
        email,
        password,
      });

      console.log("Login Response:", res.data);

      if (res.data && res.data.email) {
        localStorage.setItem("userEmail", res.data.email);
        navigate("/resume");
      } else {
        alert("Invalid credentials! Redirecting to signup.");
        navigate("/signup");
      }
    } catch (error) {
      console.error("Login failed:", error.response?.data || error.message);
      alert("Invalid credentials! Redirecting to signup.");
      navigate("/signup");
    }
  };

  // Handle Google Login
  const handleLoginSuccess = async (credentialResponse) => {
    try {
      const res = await axios.post("http://localhost:8080/api/users/login", {
        token: credentialResponse.credential,
      });

      console.log("Backend Response:", res.data);

      if (res.data.email) {
        localStorage.setItem("userEmail", res.data.email);
        await fetchUserProfile(res.data.email);
        navigate("/resume");
      } else {
        alert("Google login failed!");
      }
    } catch (error) {
      console.error("OAuth login failed:", error.response?.data || error.message);
      alert("Google login failed!");
    }
  };

  // Fetch User Profile
  const fetchUserProfile = async (email) => {
    try {
      const res = await axios.get(`http://localhost:8080/api/users/me?email=${email}`);

      console.log("User Profile:", res.data);
      if (res.status === 200 && res.data) {
        setUser(res.data);
      } else {
        console.error("User profile not found");
      }
    } catch (error) {
      console.error("Error fetching user profile:", error);
    }
  };

  // Auto-fetch profile if user is already logged in
  useEffect(() => {
    const storedEmail = localStorage.getItem("userEmail");
    if (storedEmail) {
      fetchUserProfile(storedEmail);
    }
  }, []);

  return (
    <GoogleOAuthProvider clientId={clientId}>
      <div>
        <nav className="Nav">
          <img src="logo.png" alt="Logo" />
          <h1 className="Inrovise">Introvise</h1>
        </nav>

        <div className="half">
            <div className="slider">
              <Slider {...settings}>
            <div>
              <img className="slider-img" src="/cards1.png" alt="Abstract 1" />
            </div>
            <div>
              <img className="slider-img" src="/cards2.png" alt="Abstract 2" />
            </div>
            <div>
              <img className="slider-img" src="/cards3.png" alt="Abstract 3" />
            </div>
            </Slider>
        </div>
        </div>

        <p className="head">
          Welcome Back to <span>Introvise!</span>
        </p>
        <p className="sec">Sign in with </p>

        {/* Google Login Button */}
        <div className="google-login1">
          <GoogleLogin
            onSuccess={handleLoginSuccess}
            onError={() => alert("Google Login Failed")}
          />
        </div>

        <span className="border">
          <p className="text"> OR </p>
        </span>

        <p className="E">Email</p>
        <input
          className="Email"
          type="text"
          placeholder="Enter Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />

        <p className="P">Password</p>
        <input
          className="Pass"
          type="password"
          placeholder="Enter Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <button className="btnSign" onClick={handleSignIn}>
          Sign In
        </button>
        <a href="#" className="a">
          Forgot password?
        </a>

        <p className="np">
          Don&lsquo;t have an account? <a href="/signup">Sign Up</a>
        </p>
      </div>
    </GoogleOAuthProvider>
  );
}

export default Login;
