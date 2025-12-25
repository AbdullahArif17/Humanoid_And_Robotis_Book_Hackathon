import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import ChatInterface from '../components/ChatInterface';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Read the Book
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();

  return (
    <Layout
      title={`AI-Native Book: Physical AI & Humanoid Robotics`}
      description="Comprehensive guide to humanoid robotics combining AI with physical systems">
      <HomepageHeader />
      <main>
        {/* Hero Section */}
        <section className="padding-top--xl padding-bottom--lg">
          <div className="container">
            <div className="row">
              <div className="col col--8 col--offset-2">
                <div className="text--center padding-horiz--md">
                  <h1 className="hero-title">Physical AI & Humanoid Robotics</h1>
                  <p className="hero-subtitle">From Digital Intelligence to Embodied Systems</p>
                  <div className="hero-actions margin-top--lg">
                    <Link
                      className="button button--primary button--lg"
                      to="/docs/intro">
                      Start Learning
                    </Link>
                    <Link
                      className="button button--secondary button--lg margin-left--md"
                      to="/docs/module-1-ros2/introduction">
                      Explore Modules
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Features Grid */}
        <section className="padding-top--lg padding-bottom--xl">
          <div className="container">
            <div className="text--center margin-bottom--lg">
              <h2>Core Modules</h2>
              <p className="hero-subtitle">Comprehensive coverage of humanoid robotics fundamentals</p>
            </div>

            <div className="row">
              <div className="col col--3">
                <div className="feature-card padding--md">
                  <h3>ROS 2 Fundamentals</h3>
                  <p>Learn the foundation of robotic systems programming with ROS 2 for humanoid applications.</p>
                  <Link to="/docs/module-1-ros2/introduction" className="feature-link">
                    Explore →
                  </Link>
                </div>
              </div>

              <div className="col col--3">
                <div className="feature-card padding--md">
                  <h3>Simulation & Control</h3>
                  <p>Master Gazebo and Unity for humanoid robotics simulation and advanced control techniques.</p>
                  <Link to="/docs/module-2-simulation/introduction" className="feature-link">
                    Explore →
                  </Link>
                </div>
              </div>

              <div className="col col--3">
                <div className="feature-card padding--md">
                  <h3>NVIDIA Isaac Platform</h3>
                  <p>Advanced AI for humanoid robotics using NVIDIA Isaac platform and Isaac Sim.</p>
                  <Link to="/docs/module-3-nvidia-isaac/introduction" className="feature-link">
                    Explore →
                  </Link>
                </div>
              </div>

              <div className="col col--3">
                <div className="feature-card padding--md">
                  <h3>Vision-Language-Action</h3>
                  <p>Vision-Language-Action systems for advanced humanoid capabilities and perception.</p>
                  <Link to="/docs/module-4-vla/introduction" className="feature-link">
                    Explore →
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* AI Chatbot Section */}
        <section className="padding-top--xl padding-bottom--xl" style={{ background: 'var(--neutral-50)' }}>
          <div className="container">
            <div className="row">
              <div className="col col--8 col--offset-2">
                <div className="text--center margin-bottom--lg">
                  <h2>AI-Powered Assistant</h2>
                  <p>Ask questions about the book content using our AI-powered chatbot with Retrieval-Augmented Generation (RAG).</p>
                </div>

                <div className="chat-container-wrapper" style={{
                  height: '500px',
                  background: 'white',
                  border: '1px solid var(--neutral-200)',
                  borderRadius: '12px',
                  overflow: 'hidden',
                  boxShadow: '0 10px 25px rgba(0, 0, 0, 0.05)'
                }}>
                  <ChatInterface apiUrl="https://abdullah017-humanoid-and-robotis-book.hf.space" />
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Call to Action */}
        <section className="padding-top--xl padding-bottom--xl">
          <div className="container">
            <div className="row">
              <div className="col col--8 col--offset-2">
                <div className="text--center">
                  <h2>Ready to Build Humanoid Robots?</h2>
                  <p className="margin-bottom--lg">Start your journey in physical AI and humanoid robotics today.</p>
                  <Link
                    className="button button--primary button--lg"
                    to="/docs/intro">
                    Begin Your Learning Path
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