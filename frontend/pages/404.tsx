import React from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { motion } from 'framer-motion';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { Home, ArrowLeft, Search } from 'lucide-react';

export default function Custom404() {
  return (
    <>
      <Head>
        <title>Page Not Found - ResuMatch</title>
        <meta name="description" content="The page you're looking for doesn't exist." />
      </Head>

      <div className="min-h-screen bg-gray-50">
        <Header />
        
        <main className="pt-24 pb-16">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="mb-12"
            >
              <div className="text-8xl md:text-9xl font-bold gradient-text mb-8">
                404
              </div>
              <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
                Page Not Found
              </h1>
              <p className="text-xl text-gray-600 max-w-2xl mx-auto mb-12">
                Sorry, we couldn't find the page you're looking for. It might have been moved, 
                deleted, or you might have entered the wrong URL.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Link href="/" className="btn-primary inline-flex items-center">
                    <Home className="w-5 h-5 mr-2" />
                    Go Home
                  </Link>
                </motion.div>
                
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <button 
                    onClick={() => window.history.back()} 
                    className="btn-secondary inline-flex items-center"
                  >
                    <ArrowLeft className="w-5 h-5 mr-2" />
                    Go Back
                  </button>
                </motion.div>
                
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Link href="/help" className="btn-secondary inline-flex items-center">
                    <Search className="w-5 h-5 mr-2" />
                    Get Help
                  </Link>
                </motion.div>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="card p-8"
            >
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Quick Links
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-left">
                <Link 
                  href="/analyze" 
                  className="p-4 rounded-lg border hover:border-primary-300 hover:bg-primary-50 transition-colors duration-200"
                >
                  <h3 className="font-medium text-gray-900">Analyze Resume</h3>
                  <p className="text-sm text-gray-600">Upload and analyze your resume</p>
                </Link>
                
                <Link 
                  href="/about" 
                  className="p-4 rounded-lg border hover:border-primary-300 hover:bg-primary-50 transition-colors duration-200"
                >
                  <h3 className="font-medium text-gray-900">About ResuMatch</h3>
                  <p className="text-sm text-gray-600">Learn about our platform</p>
                </Link>
                
                <Link 
                  href="/help" 
                  className="p-4 rounded-lg border hover:border-primary-300 hover:bg-primary-50 transition-colors duration-200"
                >
                  <h3 className="font-medium text-gray-900">Help & Support</h3>
                  <p className="text-sm text-gray-600">Find answers to common questions</p>
                </Link>
              </div>
            </motion.div>
          </div>
        </main>

        <Footer />
      </div>
    </>
  );
}
