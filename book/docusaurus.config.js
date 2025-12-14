// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'AI-Native Book: Physical AI & Humanoid Robotics',
  tagline: 'Comprehensive guide to humanoid robotics combining AI with physical systems',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://AbdullahArif17.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/Humanoid_And_Robotis_Book_Hackathon/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'AbdullahArif17', // Usually your GitHub org/user name.
  projectName: 'Humanoid_And_Robotis_Book_Hackathon', // Usually your repo name.
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/AbdullahArif17/Humanoid_And_Robotis_Book_Hackathon/edit/main/book/',
        },
        blog: false, // Optional: disable the blog plugin
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Humanoid Robotics Book',
        logo: {
          alt: 'Humanoid Robotics Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Book',
          },
          {
            href: 'https://github.com/AbdullahArif17/Humanoid_And_Robotis_Book_Hackathon',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Modules',
            items: [
              {
                label: 'Module 1: ROS 2 for Humanoid Robotics',
                to: '/docs/module-1-ros2/introduction',
              },
              {
                label: 'Module 2: Gazebo & Unity for Humanoid Simulation',
                to: '/docs/module-2-simulation/introduction',
              },
              {
                label: 'Module 3: NVIDIA Isaac for Humanoid AI',
                to: '/docs/module-3-nvidia-isaac/introduction',
              },
              {
                label: 'Module 4: Vision-Language-Action for Humanoid Robotics',
                to: '/docs/module-4-vla/introduction',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/humanoid-robotics',
              },
              {
                label: 'Discord',
                href: 'https://discordapp.com/invite/humanoid-robotics',
              },
              {
                label: 'Twitter',
                href: 'https://twitter.com/humanoidrobotics',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/AbdullahArif17/Humanoid_And_Robotis_Book_Hackathon',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} AI-Native Book: Physical AI & Humanoid Robotics. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;