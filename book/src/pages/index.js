import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import EmbeddedChatInterface from '../components/ChatInterface/EmbeddedChatInterface';

import styles from './index.module.css';

const technologies = [
  { name: 'ROS 2', icon: '🤖' },
  { name: 'NVIDIA Isaac', icon: '🧠' },
  { name: 'Gazebo', icon: '🔥' },
  { name: 'Unity', icon: '🎮' },
  { name: 'Python', icon: '🐍' },
  { name: 'C++', icon: '⚙️' },
];

const stats = [
  { number: '4', label: 'Core Modules' },
  { number: '20+', label: 'Chapters' },
  { number: '50+', label: 'Topics' },
  { number: '∞', label: 'Possibilities' },
];

const features = [
  {
    title: 'ROS 2 Fundamentals',
    icon: '🤖',
    description: 'Learn the foundation of robotic systems programming with ROS 2 for humanoid applications.',
    link: '/docs/module-1-ros2/introduction',
  },
  {
    title: 'Simulation & Control',
    icon: '🎮',
    description: 'Master Gazebo and Unity for humanoid robotics simulation and advanced control techniques.',
    link: '/docs/module-2-simulation/introduction',
  },
  {
    title: 'NVIDIA Isaac Platform',
    icon: '🧠',
    description: 'Advanced AI for humanoid robotics using NVIDIA Isaac platform and Isaac Sim.',
    link: '/docs/module-3-nvidia-isaac/introduction',
  },
  {
    title: 'Vision-Language-Action',
    icon: '👁️',
    description: 'VLA systems for advanced humanoid capabilities and perception.',
    link: '/docs/module-4-vla/introduction',
  },
];

function FeatureCard({ title, icon, description, link }) {
  return (
    <div className="col col--3">
      <Link to={link} style={{ textDecoration: 'none', color: 'inherit' }}>
        <div className="glass-card padding--lg" style={{ height: '100%', borderRadius: '24px' }}>
          <div className="feature-icon">
            <span>{icon}</span>
          </div>
          <h3 style={{ marginTop: 0, fontSize: '1.25rem' }}>{title}</h3>
          <p style={{ color: 'var(--neutral-600)', fontSize: '0.95rem' }}>{description}</p>
          <div className="feature-link" style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            Learn More <span style={{ transition: 'transform 0.2s' }}>→</span>
          </div>
        </div>
      </Link>
    </div>
  );
}

function RoadmapItem({ number, title, chapters, description, align }) {
  return (
    <div className={styles.roadmapItem}>
      <div className={clsx(styles.roadmapContent, 'glass-card fade-in-section')}>
        <span className={styles.roadmapNumber}>{number}</span>
        <h3 className="gradient-text" style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>{title}</h3>
        <p style={{ fontWeight: 600, color: 'var(--primary-600)', marginBottom: '0.5rem' }}>{chapters}</p>
        <p style={{ color: 'var(--neutral-600)', margin: 0 }}>{description}</p>
      </div>
      <div className={styles.roadmapDot}></div>
      <div className={styles.roadmapContent} style={{ visibility: 'hidden' }}></div>
    </div>
  );
}

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner, 'hero-gradient-bg')}>
      <div className="floating-element" style={{ top: '15%', left: '8%', fontSize: '4rem' }}>🤖</div>
      <div className="floating-element" style={{ top: '25%', right: '12%', fontSize: '3rem' }}>🚀</div>
      <div className="floating-element" style={{ bottom: '20%', left: '18%', fontSize: '3.5rem' }}>🦾</div>
      
      <div className="container" style={{ position: 'relative', zIndex: 1 }}>
        <div className="badge margin-bottom--md" style={{ 
          display: 'inline-block',
          background: 'rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(10px)',
          padding: '0.5rem 1rem',
          borderRadius: '100px',
          color: 'white',
          fontSize: '0.875rem',
          fontWeight: 600,
          border: '1px solid rgba(255, 255, 255, 0.2)'
        }}>
          Powered by Physical AI & Robotics
        </div>
        <h1 className={clsx('hero__title', styles.heroTitle)} style={{ letterSpacing: '-0.02em' }}>
          {siteConfig.title}
        </h1>
        <p className={clsx('hero__subtitle', styles.heroSubtitle)} style={{ maxWidth: '800px', margin: '0 auto 2.5rem' }}>
          {siteConfig.tagline}
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--lg button-glow"
            style={{ 
              background: 'white', 
              color: 'var(--primary-600)',
              padding: '1rem 2.5rem',
              fontSize: '1.1rem'
            }}
            to="/docs/intro">
            📚 Start Reading
          </Link>
          <Link
            className="button button--secondary button--lg"
            style={{ 
              color: 'white',
              borderColor: 'rgba(255, 255, 255, 0.3)',
              padding: '1rem 2.5rem',
              fontSize: '1.1rem'
            }}
            to="https://github.com/AbdullahArif17/Humanoid_And_Robotis_Book_Hackathon">
            ⭐️ View on GitHub
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();

  React.useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
          }
        });
      },
      { threshold: 0.1 }
    );

    const sections = document.querySelectorAll('.fade-in-section');
    sections.forEach((section) => observer.observe(section));

    return () => {
      sections.forEach((section) => observer.unobserve(section));
    };
  }, []);

  return (
    <Layout
      title={`AI-Native Book: Physical AI & Humanoid Robotics`}
      description="Comprehensive guide to humanoid robotics combining AI with physical systems">
      <HomepageHeader />
      <main>
        {/* Stats Section */}
        <section className="padding-bottom--lg">
          <div className="container">
            <div className={styles.statsSection}>
              {stats.map((stat, idx) => (
                <div key={idx} className="stat-item">
                  <div className="stat-number">{stat.number}</div>
                  <div className="stat-label">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Hero Content Section */}
        <section className="padding-top--xl padding-bottom--xl fade-in-section">
          <div className="container">
            <div className="row align-items--center">
              <div className="col col--6">
                <h2 className="gradient-text" style={{ textAlign: 'left', fontSize: '2.5rem', marginBottom: '1.5rem' }}>
                  The Future of Embodied Intelligence
                </h2>
                <p style={{ fontSize: '1.15rem', color: 'var(--neutral-600)', marginBottom: '2rem' }}>
                  This book provides a comprehensive journey from the mathematical foundations of robotics to the cutting edge of physical AI. 
                  Learn how to build, simulate, and deploy humanoid robots using industry-standard tools and frameworks.
                </p>
                <div className="hero-actions" style={{ display: 'flex', gap: '1rem' }}>
                  <Link className="button button--primary button--lg button-glow" to="/docs/intro">
                    Begin Journey
                  </Link>
                </div>
              </div>
              <div className="col col--6">
                <div className="glass-card padding--xl text--center" style={{ borderRadius: '32px', position: 'relative', overflow: 'hidden' }}>
                  <div style={{ fontSize: '10rem', opacity: 0.1, position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}>🤖</div>
                  <div style={{ position: 'relative', zIndex: 1 }}>
                    <h3 style={{ fontSize: '2rem', marginBottom: '1rem' }}>AI-Native Learning</h3>
                    <p style={{ color: 'var(--neutral-600)' }}>
                      Every chapter is designed with AI implementation in mind, featuring code examples, simulation guides, and theoretical depth.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Roadmap Section */}
        <section className="padding-top--xl padding-bottom--xl" style={{ background: 'var(--neutral-50)' }}>
          <div className="container">
            <div className="text--center margin-bottom--xl">
              <h2 style={{ fontSize: '2.5rem' }}>Learning Roadmap</h2>
              <div className="premium-divider"></div>
              <p className="hero-subtitle">Your structured path to mastering humanoid robotics</p>
            </div>

            <div className={styles.roadmapContainer}>
              <div className={styles.roadmapPath}></div>
              <RoadmapItem 
                number="01"
                title="ROS 2 Fundamentals"
                chapters="Chapters 1-5"
                description="Master the middleware of modern robotics. Learn nodes, topics, services, and the robotic ecosystem."
              />
              <RoadmapItem 
                number="02"
                title="Simulation & Dynamics"
                chapters="Chapters 6-12"
                description="High-fidelity simulation in Gazebo and Unity. URDF modeling and humanoid kinematics."
              />
              <RoadmapItem 
                number="03"
                title="NVIDIA Isaac Sim"
                chapters="Chapters 13-18"
                description="Leverage GPU-accelerated simulation and reinforcement learning for physical AI."
              />
              <RoadmapItem 
                number="04"
                title="VLA & Perception"
                chapters="Chapters 19-25"
                description="Vision-Language-Action models and advanced perception for autonomous humanoid behavior."
              />
            </div>
          </div>
        </section>

        {/* Features Grid */}
        <section className="padding-top--xl padding-bottom--xl">
          <div className="container">
            <div className="text--center margin-bottom--xl">
              <h2 style={{ marginBottom: '0.5rem' }}>Core Modules</h2>
              <div className="premium-divider"></div>
              <p className="hero-subtitle">Comprehensive coverage of humanoid robotics fundamentals</p>
            </div>

            <div className="row" style={{ gap: '1.5rem 0' }}>
              {features.map((feature, idx) => (
                <FeatureCard key={idx} {...feature} />
              ))}
            </div>
          </div>
        </section>

        {/* AI Chatbot Section */}
        <section className="padding-top--xl padding-bottom--xl fade-in-section" id="ai-assistant">
          <div className="container">
            <div className="row">
              <div className="col col--10 col--offset-1">
                <div className="text--center margin-bottom--lg">
                  <h2 className="gradient-text" style={{ marginBottom: '0.5rem' }}>🤖 Intelligent Assistant</h2>
                  <div className="premium-divider"></div>
                  <p className="hero-subtitle">Ask anything about the book content, from ROS 2 setup to VLA architecture</p>
                </div>

                <div className={styles.aiChatPremiumContainer}>
                  <div className={clsx("glass-card", styles.chatWrapper)} style={{
                    height: '600px',
                    borderRadius: '24px',
                    overflow: 'hidden',
                    border: '1px solid rgba(59, 130, 246, 0.2)',
                    boxShadow: '0 20px 50px rgba(0, 0, 0, 0.1)',
                  }}>
                    <EmbeddedChatInterface apiUrl="https://abdullah017-humanoid-and-robotis-book.hf.space" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Tech Stack Section */}
        <section className="padding-top--xl padding-bottom--xl">
          <div className="container">
            <div className="text--center margin-bottom--xl">
              <h2 style={{ fontSize: '2rem' }}>Core Technologies</h2>
              <div className="premium-divider"></div>
            </div>
            <div className={styles.techGrid}>
              {technologies.map((tech, idx) => (
                <div key={idx} className={styles.techItem}>
                  <span className={styles.techIcon}>{tech.icon}</span>
                  <span style={{ fontWeight: 600, color: 'var(--neutral-600)' }}>{tech.name}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Final CTA */}
        <section className="padding-top--xl padding-bottom--xl">
          <div className="container">
            <div className={clsx(styles.ctaSection, "text--center")}>
              <h2 style={{ color: 'white', fontSize: '3rem', marginBottom: '1.5rem' }}>The Future is Humanoid</h2>
              <p className="margin-bottom--lg" style={{ color: 'rgba(255, 255, 255, 0.8)', maxWidth: '600px', margin: '0 auto 2.5rem', fontSize: '1.25rem' }}>
                Join the revolution in physical AI. Start your journey into the world of humanoid robotics today.
              </p>
              <Link
                className="button button--primary button--lg button-glow"
                style={{ background: 'white', color: 'var(--primary-600)', padding: '1.25rem 3rem' }}
                to="/docs/intro">
                Get Started Now 🎯
              </Link>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}