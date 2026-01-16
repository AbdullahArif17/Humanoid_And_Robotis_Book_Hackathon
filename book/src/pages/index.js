import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import EmbeddedChatInterface from '../components/ChatInterface/EmbeddedChatInterface';

import styles from './index.module.css';

// Feature data with icons
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

// Stats data
const stats = [
  { number: '4', label: 'Core Modules' },
  { number: '20+', label: 'Chapters' },
  { number: '50+', label: 'Topics' },
  { number: '∞', label: 'Possibilities' },
];

function FeatureCard({ title, icon, description, link }) {
  return (
    <div className="col col--3">
      <div className="glass-card padding--lg" style={{ height: '100%', borderRadius: '16px' }}>
        <div className="feature-icon">
          <span>{icon}</span>
        </div>
        <h3 style={{ marginTop: 0 }}>{title}</h3>
        <p style={{ color: 'var(--neutral-600)' }}>{description}</p>
        <Link to={link} className="feature-link">
          Explore →
        </Link>
      </div>
    </div>
  );
}

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner, 'hero-gradient-bg')}>
      {/* Floating decorative elements */}
      <div className="floating-element" style={{ top: '10%', left: '5%', fontSize: '4rem' }}>🤖</div>
      <div className="floating-element" style={{ top: '20%', right: '10%', fontSize: '3rem' }}>⚙️</div>
      <div className="floating-element" style={{ bottom: '15%', left: '15%', fontSize: '3.5rem' }}>🧠</div>
      
      <div className="container" style={{ position: 'relative', zIndex: 1 }}>
        <h1 className={clsx('hero__title', styles.heroTitle)}>
          {siteConfig.title}
        </h1>
        <p className={clsx('hero__subtitle', styles.heroSubtitle)}>
          {siteConfig.tagline}
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--lg button-glow"
            style={{ 
              background: 'white', 
              color: 'var(--primary-600)',
              fontWeight: 600,
              border: 'none'
            }}
            to="/docs/intro">
            📚 Read the Book
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
        <section className="padding-top--xl padding-bottom--lg">
          <div className="container">
            <div className="stats-section">
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
        <section className="padding-top--lg padding-bottom--lg fade-in-section">
          <div className="container">
            <div className="row">
              <div className="col col--8 col--offset-2">
                <div className="text--center padding-horiz--md">
                  <h2 className="gradient-text" style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>
                    Physical AI & Humanoid Robotics
                  </h2>
                  <div className="premium-divider"></div>
                  <p className="hero-subtitle" style={{ maxWidth: '600px', margin: '0 auto' }}>
                    From Digital Intelligence to Embodied Systems — Master the technologies shaping the future of robotics
                  </p>
                  <div className="hero-actions margin-top--lg" style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
                    <Link
                      className="button button--primary button--lg button-glow"
                      to="/docs/intro">
                      🚀 Start Learning
                    </Link>
                    <Link
                      className="button button--secondary button--lg"
                      to="/docs/module-1-ros2/introduction">
                      📖 Explore Modules
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Features Grid */}
        <section className="padding-top--xl padding-bottom--xl" style={{ background: 'var(--neutral-50)' }}>
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
        <section className="padding-top--xl padding-bottom--xl fade-in-section">
          <div className="container">
            <div className="row">
              <div className="col col--10 col--offset-1">
                <div className="text--center margin-bottom--lg">
                  <h2 className="gradient-text" style={{ marginBottom: '0.5rem' }}>🤖 Intelligent Assistant</h2>
                  <div className="premium-divider"></div>
                  <p className="hero-subtitle">Interactive RAG-powered chatbot trained on the complete book content</p>
                </div>

                <div className="ai-chat-premium-container">
                  <div className="glass-card chat-wrapper" style={{
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

        {/* Call to Action */}
        <section className="padding-top--xl padding-bottom--xl">
          <div className="container">
            <div className="row">
              <div className="col col--10 col--offset-1">
                <div className="cta-section text--center">
                  <h2 style={{ marginBottom: '0.5rem' }}>Ready to Build Humanoid Robots?</h2>
                  <div className="premium-divider"></div>
                  <p className="margin-bottom--lg" style={{ maxWidth: '500px', margin: '0 auto 2rem' }}>
                    Start your journey in physical AI and humanoid robotics today. Join thousands of learners exploring the future of robotics.
                  </p>
                  <Link
                    className="button button--primary button--lg button-glow"
                    to="/docs/intro">
                    🎯 Begin Your Learning Path
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}