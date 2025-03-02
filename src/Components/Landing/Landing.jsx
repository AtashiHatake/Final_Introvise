import { useEffect, useState } from "react";
import { FaRegPlayCircle } from "react-icons/fa";
import './Landing.css';

const TestimonialCarousel = () => {
  const testimonials = [
    {
      company: "Zoomer",
      text: "Non risus viverra enim, quis. Eget vitae arcu vivamus sit tellus, viverra turpis lorem. Varius a turpis urna id porttitor.",
      name: "Helena Jummy",
      role: "Team Lead"
    },
    {
      company: "SHELLS",
      text: "Aliquet ridiculus mi porta habitant vulputate rhoncus, mattis amet enim. Sit purus venenatis velit semper lectus sed ornare quam nulla.",
      name: "Helena John",
      role: "Co-founder"
    },
    {
      company: "ArtVenue",
      text: "A eget sed posuere dui risus habitasse mauris. Venenatis aliquet id ultrices a lacus. Pretium vehicula pretium posuere justo sed lorem cursus.",
      name: "David Oshodi",
      role: "Manager"
    },
    {
      company: "WAVES",
      text: "Magna egestas aliquet ut integer non. Sed diam enim nibh sit. Aliquam laoreet aenean metus nibh eu scelerisque.",
      name: "Charolotte Hanlin",
      role: "CEO"
    }
  ];

  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveIndex(prev => (prev + 1) % testimonials.length);
    }, 2000); 
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="testimonial-section">
      <h1 className="hidden slide-left">What everyone says</h1>
      <div className="carousel-container">
        <div 
          className="carousel-track"
          style={{ 
            transform: `translateX(-${activeIndex * 100}%)`,
            width: `${testimonials.length * 100}%`,
            transition: activeIndex === 0 ? 'none' : 'transform 500ms ease-in-out'
          }}
        >
          {testimonials.map((testimonial, index) => (
            <div key={index} className="testimonial-card-wrapper">
              <div className={`testimonial-card ${index === activeIndex ? 'active' : ''}`}>
                <h3>{testimonial.company}</h3>
                <p>{testimonial.text}</p>
                <div className="testimonial-footer">
                  <p className="name">{testimonial.name}</p>
                  <p className="role">{testimonial.role}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

function Landing() {
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("show");
          } else {
            entry.target.classList.remove("show");
          }
        });
      },
      { threshold: 0.2 }
    );

    const elements = document.querySelectorAll(".hidden");
    elements.forEach((el) => observer.observe(el));

    return () => observer.disconnect();
  }, []);

  return (
    <div>
      <nav className="navbar">
        <div className="logo">Introvise</div>
        <div className="nav-middle">
          <a href="/">Home</a>
          <a href="/interviews">Interviews</a>
          <a href="/about">About Us</a>
          <a href="/use-cases">Use Cases</a>
        </div>
        <div className="nav-right">
          <a id="login-l" href="/login">Log in</a>
          <a href="/signup" className="signup-button">Sign Up Now</a>
        </div>
      </nav>  

      <div className="App">
        <img className="img-sec" src="PS.svg" alt="PS" />
        <button className="masked-button">Sign Up</button> 
        <button className="masked-button2">
          <FaRegPlayCircle className="play-icon" /> View Demo
        </button>
      </div>

      <div className="pg1">
        <h1 className="hidden slide-left">Features</h1>
        <p className="hidden slide-left">
          Explore the cutting-edge features that Introvise offers to enhance your <br />
          interview preparation and performance.
        </p>
        <div className="image-container">
          <img className="img-l hidden slide-up" src="FC.svg" alt="Feature 1" />
          <img className="img-l hidden slide-up" src="FC1.svg" alt="Feature 2" />
          <img className="img-l hidden slide-up" src="FC2.svg" alt="Feature 3" />
        </div>
      </div>

      <div className="pg2">
        <h1 className="hidden slide-left">How It Works</h1>
        <p className="hidden slide-left">Practice, analyze, improve - all with AI-driven insights.</p>
      </div>
      
      <div className="pg3">
        <img className="img-l2" src="1.svg" alt="1.svg" />
        <img className="img-l2" src="2.svg" alt="2.svg" />
        <img className="img-l2" src="3.svg" alt="3.svg" /> 
        <div>
          <img className="pg3" src="pg3.svg" alt="pg3.svg" />
        </div>
      </div>
      
      <TestimonialCarousel />

      <div> 
        <img className="pg4" src="CTA.svg" alt="CTA.svg" />
        <img className="pg4" src="CTA-B.svg" alt="" />
      </div>
      <div className="footer">
       <img src="Container.svg" alt="Container.svg" />
      </div>
    </div>
  );
}

export default Landing;
