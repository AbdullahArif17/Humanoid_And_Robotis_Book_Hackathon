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
      title={`Hello from ${siteConfig.title}`}
      description="AI-Native Book on Physical AI & Humanoid Robotics with integrated RAG Chatbot">
      <HomepageHeader />
      <main>
        <section className={styles.modulesSection}>
          <div className="container">
            <h2>Book Modules</h2>
            <div className="row">
              <div className="col col--3">
                <div className="text--center padding-horiz--md">
                  <h3>Module 1: ROS 2</h3>
                  <p>Learn about Robot Operating System 2 for humanoid robotics applications</p>
                </div>
              </div>
              <div className="col col--3">
                <div className="text--center padding-horiz--md">
                  <h3>Module 2: Simulation</h3>
                  <p>Gazebo and Unity for humanoid robotics simulation</p>
                </div>
              </div>
              <div className="col col--3">
                <div className="text--center padding-horiz--md">
                  <h3>Module 3: NVIDIA Isaac</h3>
                  <p>Advanced AI for humanoid robotics using NVIDIA Isaac platform</p>
                </div>
              </div>
              <div className="col col--3">
                <div className="text--center padding-horiz--md">
                  <h3>Module 4: VLA</h3>
                  <p>Vision-Language-Action systems for advanced humanoid capabilities</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.chatSection}>
          <div className="container padding-vert--lg">
            <h2>AI-Powered Chatbot</h2>
            <p>Ask questions about the book content using our AI-powered chatbot with Retrieval-Augmented Generation (RAG).</p>
            <div style={{ height: '500px', border: '1px solid #ccc', borderRadius: '8px', overflow: 'hidden' }}>
              <ChatInterface apiUrl="https://abdullah017-humanoid-and-robotis-book.hf.space" />
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}