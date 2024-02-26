import React, { Component } from 'react';
import { Divider, Form, Input, Button, Segment, Message, Select } from 'semantic-ui-react';
import Layout from '../components/Layout';
import record from '../ethereum/record';
import web3 from '../ethereum/web3';



class Diagnosis extends Component {
    render() {
        return (
            <Layout>
                <Segment padded><h1>Brain Tumor Analysis</h1></Segment>
                <Segment>
                <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".png, .jpg, .jpeg" />
                <button type="submit">Predict</button>
                </form>
                <Divider clearing />
               
                </Segment>
            </Layout>
        );
    }
}

export default Diagnosis;