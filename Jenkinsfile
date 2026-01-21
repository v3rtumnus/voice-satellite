pipeline {
    agent any
    
    triggers {
        pollSCM('H/5 * * * *')  // Check every 5 minutes
    }
    
    environment {
        // Comma-separated list of target nodes
        TARGET_NODES = "${env.VOICE_SATELLITE_NODES ?: '192.168.1.100,192.168.1.101'}"
        
        // SSH credentials ID configured in Jenkins
        SSH_CREDENTIALS_ID = 'satellite-ssh-key'
        
        // Remote user and paths
        REMOTE_USER = "${env.VOICE_SATELLITE_USER ?: 'pi'}"
        REMOTE_DIR = '/opt/voice-satellite'
        
        // Directories to copy (relative to repo root)
        COPY_DIRS = 'satellite,scripts,sounds'
        
        // Service to restart
        SERVICE_NAME = 'voice-satellite'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Deploy to Nodes') {
            steps {
                script {
                    def nodes = TARGET_NODES.split(',')
                    def copyDirs = COPY_DIRS.split(',')
                    
                    // Deploy to each node in parallel
                    def deployments = [:]
                    
                    nodes.each { node ->
                        def nodeIp = node.trim()
                        deployments["Deploy to ${nodeIp}"] = {
                            stage("Deploy to ${nodeIp}") {
                                withCredentials([sshUserPrivateKey(
                                    credentialsId: SSH_CREDENTIALS_ID,
                                    keyFileVariable: 'SSH_KEY',
                                    usernameVariable: 'SSH_USER'
                                )]) {
                                    // Copy each directory
                                    copyDirs.each { dir ->
                                        def dirName = dir.trim()
                                        sh """
                                            echo "Copying ${dirName} to ${nodeIp}:${REMOTE_DIR}/"
                                            scp -i \$SSH_KEY -o StrictHostKeyChecking=no -r \
                                                ${dirName} \
                                                ${REMOTE_USER}@${nodeIp}:${REMOTE_DIR}/
                                        """
                                    }
                                    
                                    // Restart the service
                                    sh """
                                        echo "Restarting ${SERVICE_NAME} on ${nodeIp}"
                                        ssh -i \$SSH_KEY -o StrictHostKeyChecking=no \
                                            ${REMOTE_USER}@${nodeIp} \
                                            'sudo systemctl restart ${SERVICE_NAME}'
                                    """
                                }
                            }
                        }
                    }
                    
                    parallel deployments
                }
            }
        }
    }
    
    post {
        success {
            echo "Deployment successful to all nodes: ${TARGET_NODES}"
        }
        failure {
            echo "Deployment failed!"
        }
    }
}
